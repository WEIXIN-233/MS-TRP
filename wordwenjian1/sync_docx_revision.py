# -*- coding: utf-8 -*-
"""把 LaTeX 稿（main_ctexart.tex）这一轮评审修订同步到 Word 稿。

只做定点替换/插入：
  · 正文段落按索引整段替换（保留段落样式与首 run 字体）；
  · 新增段落用现有段落 XML 深拷贝插入，保持版式；
  · §4.2 的式(7) 按修正后的对称归一化重写 OMML；
  · 新增 §6.9（含表11）与 §6.10。
运行前会自动备份为 *_同步修订前_YYYYmmdd-HHMMSS.docx。
"""
import copy
import datetime
import json
import os
import re
import shutil

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = os.environ.get("MSTRP_DOCX", os.path.join(HERE, "MS-TRP_中文双栏论文 .docx"))
PERF = os.environ.get("MSTRP_PERF",
                      os.path.join(os.path.dirname(HERE), "result_raw", "perf_profile.json"))


def set_text(p, text):
    """保留首个 run 的格式，整段换成 text。"""
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def clone_after(anchor, template, text):
    """在 anchor 段后插入一个与 template 同格式的新段。"""
    el = copy.deepcopy(template._p)
    anchor._p.addnext(el)
    from docx.text.paragraph import Paragraph
    p = Paragraph(el, anchor._parent)
    set_text(p, text)
    return p


M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _r(t):
    return f'<m:r><m:t xml:space="preserve">{t}</m:t></m:r>'


def _sup(base, sup):
    return (f'<m:sSup><m:e>{_r(base)}</m:e><m:sup>{sup}</m:sup></m:sSup>')


def _sub(base, sub):
    return (f'<m:sSub><m:e>{_r(base)}</m:e><m:sub>{_r(sub)}</m:sub></m:sSub>')


def _half():
    return '<m:f><m:num>' + _r("−1") + '</m:num><m:den>' + _r("2") + '</m:den></m:f>'


def eq7_omml():
    """W = D^{-1/2} A D^{-1/2},  A_{ij} = Σ_r ζ_r u_{ij,r}"""
    nary = ('<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="subSup"/>'
            '<m:supHide m:val="1"/></m:naryPr>'
            f'<m:sub>{_r("r")}</m:sub><m:sup/>'
            f'<m:e>{_sub("ζ", "r")}{_sub("u", "ij,r")}</m:e></m:nary>')
    xml = ('<m:oMath xmlns:m="%s" xmlns:w="%s">' % (M_NS, W_NS)
           + _r("W") + _r("=") + _sup("D", _half()) + _r("A") + _sup("D", _half())
           + _r(",    ")
           + _sub("A", "ij") + _r("=") + nary
           + '</m:oMath>')
    from lxml import etree
    return etree.fromstring(xml.encode("utf-8"))


# ── 替换段落（索引以当前 docx 为准）─────────────────────────────
ABSTRACT = (
    "预算执行的支付数据里，多尺度时序变化、异构实体联系与信息延迟同时存在，字段可读性也参差不齐。"
    "靠单一阈值规则，既难覆盖各类异常，也难把复核工作量压到可控范围。本文提出多尺度时序与关系风险传播模型 MS-TRP。"
    "它在七日、三十日、当季、当年四个尺度上度量金额、频次与执行进度，参照系由个体历史与同类群体共同建立；"
    "再用九类业务实体搭成阶段感知异构图，使已出现风险沿关系扩散。时序分数与关系分数经可靠度门控融合，"
    "阈值在验证集冻结，最终落到特征、历史、规则、关系四类审计证据上。数据含88,351条支付记录，"
    "2023—2024年训练、2025年验证、2026年测试。测试集上 MS-TRP 的 Payment-AUPRC 为 0.6006，"
    "高于 Isolation Forest 的 0.4143 与 Concat-IF 的 0.4008；冻结阈值下精确率 0.9447、召回率 0.1624、"
    "FAR/1000 为 1.53，属于低召回、高精度的工作点。把误报预算放宽到 10% 后，召回升到 0.6169。"
    "消融表明多尺度组合是排序性能的主要来源；阶段边权与动态门控的差异落在单次运行的波动量级内，"
    "尚不足以支持其独立价值。关系相关组件对整体 AUPRC 增益有限：实体聚集事件在冻结阈值下的召回几乎全部来自"
    "账户归属硬规则（17/43），而该规则与共享账户异常的注入方式直接对应；共享收款人子类仅覆盖 1/26。"
    "风险排序、阈值告警、事件检出要分开评价。本文结论只对所构造的合成数据成立。换一套独立生成机制后模型仍有效，"
    "只是增益收窄，仍有待真实业务数据检验。"
)

P42_BEFORE_EQ = "把各类关系的边权汇成加权邻接矩阵，再取对称归一化："

P42_AFTER_EQ = (
    "其中 D 为加权度对角矩阵。基础版对所有关系类型取 ζr = 1，即不做类型级权重混合；"
    "归一化在整个多元图上一并完成，因此 W 不是行随机矩阵，不解释为概率转移矩阵。"
    "其谱半径上界由加权度控制，配合回跳系数即可保证迭代收敛。以支付节点时序风险作为初始种子，"
    "中间业务实体初始风险为零，采用带回跳项的稀疏迭代传播[29]。"
)

P43 = (
    "时序可靠度（记 r_ts）由有效历史、特征覆盖与稳健尺度稳定性组成；关系可靠度（记 r_g）由有效邻居、"
    "关系类型覆盖与拓扑稳定性组成。各分量的标准化映射在训练阶段固定，不碰测试标签。可靠度刻画的是输入信息够不够、"
    "是否一致，不是风险大小，所以高风险对象也可以同时是高可靠度。账户一致性规则是硬规则：比对支付记录的收款人 "
    "payee_id 与其关联账户的登记归属收款人，不一致输出 s_mismatch=1，一致为 0，账户缺归属时视作一致。"
    "s_mismatch 经式(11)并入核心分数，取 1 时 s_core 直接顶到 1，形成对门控融合的硬覆盖。"
    "这条规则属于结构异常信号而非分数传播：实体聚集里的共享账户子类，靠注入把不同收款人的支付强制归并到同一账户，"
    "非归属收款人的支付就出现账户归属不一致而被直接命中。共享收款人子类不产生归属冲突，只能靠实体频次突增与评分阈值判定，"
    "第6.3节给出它在冻结阈值下仅覆盖 1/26 个事件。这个子类与规则的对应关系同时意味着，"
    "规则对共享账户子类的召回主要反映注入方式与规则定义的一致性，不能直接外推为一般关系异常的识别能力。"
)

P52 = (
    "检测器比较与阈值策略比较分开组织。前者允许每个方法在验证集上按各自规则选工作点，回答“方法整体是否可用”；"
    "后者固定同一组分数、只更换阈值，回答“阈值策略本身带来多少差别”。两套数字不合并解释。"
    "第6.3节给出一个具体例子：同一组分数在 0.6838 与 0.8432 两个冻结阈值下，实体聚集事件命中相差 10 个。"
)

P61_FEATURE = (
    "为把表示贡献与检测机制贡献分开，表4中的四个表格检测器各跑两组严格同样本输入：一组只用评分时点可观测值与"
    "预定义历史窗口聚合的原始字段，另一组在原始字段上追加 MS-TRP 的派生字段（z 偏离与 s 分数，不含规则与图信号）。"
    "追加派生表示后，ECOD、COPOD、MCD 与 CBLOF 的 Payment-AUPRC 从 0.3676、0.3894、0.3545、0.2136 变为 "
    "0.4991、0.5238、0.4517、0.2136，分别提高 0.1315、0.1343、0.0972 与 0.0000；Concat-IF 在派生表示上为 0.4008。"
    "业务表示本身确实带来可观增益，但在同等表示下这四个简单检测器仍低于 MS-TRP 的 0.6006。"
    "该对照只覆盖四个检测器，派生字段集合也不等同完整模型输入，因此只能说明表示贡献可单独观察，"
    "不能据此精确量化图分支与规则通道各自的增益。"
)

P62 = (
    "按表6的召回与测试事件数回推，六类分别命中约 157、15、7、7、3、18 个，合计 207 个命中、1,327 个真实事件，"
    "微平均召回约 0.1560。这是分层事件告警口径，不与表4目标误报阈值下的 Event-F1 直接合并。"
    "六类事件数从 16 到 1,209 不等，宏平均事件召回为 0.3680，明显高于微平均 0.1560，"
    "说明总体数字由金额突增这类大样本类主导；分类型报告可以避免少数类的小幅波动被误读为整体能力变化。"
    "各类型只有单次运行的点估计，没有重复扰动或重采样区间，本文不对类间差异作显著性判断。"
    "支付排序质量、阈值选择、事件匹配粒度，一起决定最终的事件覆盖。"
)

P63_A = (
    "实体聚集事件上，Temporal-only 召回 0.0930（4/43），只加账户一致性规则升到 0.6279（27/43），"
    "只加 PPR 扩散是 0.0465（2/43），规则与 PPR 联合落到 0.3953（17/43）。"
    "规则单独加入时 Payment-AUPRC 为 0.5078，比 Temporal-only 的 0.5734 低 0.0656，等于拿支付级排序换实体聚集覆盖。"
    "为定位规则与 PPR 联合后的回落，本文逐层核对规则命中、最终分数、冻结阈值、告警数、聚合事件与匹配命中。"
    "测试集上规则命中 24 笔支付，最终分数全部为 1.0；两个变体的冻结阈值分别为 0.6838 与 0.8432，"
    "24 笔在两种阈值下全部告警，规则命中没有被 PPR 抹掉。差别出在其余支付：+Rule 在 0.6838 阈值下有 273 个告警，"
    "+Rule+PPR 在 0.8432 阈值下有 658 个，但后者的额外告警没有覆盖前者在较低阈值下命中的共享收款人事件。"
    "固定 +Rule 的分数、只把阈值换成 0.8432，实体聚集命中由 27/43 降到 17/43；固定 +Rule+PPR 的分数、"
    "只把阈值换回 0.6838，命中升到 30/43。可见阈值重冻结本身就能造成 10 个事件的差别，PPR 在同一工作点上反而略增覆盖。"
)

P63_B = (
    "按子类拆分更清楚。43 个实体聚集事件中，共享账户 17 个、共享收款人 26 个。"
    "规则命中覆盖全部 17 个共享账户事件，在三个变体与两种阈值下都不变；共享收款人在冻结阈值下分别只命中 10、0、1 个，"
    "把误报预算放宽到 10% 后升到 17、16、17 个。冻结阈值下的实体聚集召回因此几乎全部来自账户归属硬规则："
    "17/43 由规则命中，完整模型多出的 1 个来自实体频次通道。共享收款人的结构异常在本文场景里要在更宽的工作点才可检出，"
    "不能支撑复杂关系传播具有独立价值的说法。"
)

P63_C = (
    "阶段边权固定后 AUPRC 0.6011，与完整模型 0.6006 差 0.0005；门控换成固定等权融合后 0.6029，比完整模型高 0.0023。"
    "两项差异都远小于单次运行的波动量级，且没有重复运行区间。本文据此把阶段感知边权与动态门控定位为实现细节，"
    "不作为已验证的创新点；可复现的收益主要来自多尺度时序参照与可审计的证据接口。各变体的排序性能与实体聚集召回对照见图3。"
)

P64 = (
    "在31至180天的历史组中，MS-TRP的AUPRC为0.6757，高于表8中其他方法。历史超过180天时，MS-TRP为0.6154，"
    "也高于Self/Peer的0.5697。在不超过30天的冷启动组中，MS-TRP为0.4887，略高于Peer-only的0.4819。"
    "该结果与历史充分时个体参照更具辨识力的机制相容。冷启动组中MS-TRP与纯群体参照接近，"
    "提示极低历史充分度下个体参照增益有限，但不能据此排除分组标签构成的影响。"
)

H69 = "6.9 统一阈值协议下的比较"
P69_A = (
    "为回答“精确率、召回率与事件指标的差别有多少来自阈值策略”，把全部方法放到同一条阈值规则下重评："
    "在验证集正常支付上取分数 0.99 分位（目标误报率 0.01），冻结后测试集只评一次。"
    "MS-TRP 在本节不使用分层收缩阈值，与基线走完全相同的规则。表11列出支付级结果与事件级结果。"
)
CAP11 = "表11 统一验证集阈值（正常样本0.99分位）下的测试集结果（各列加粗为最优）"
TAB11 = [
    ["方法", "Payment-AUPRC ↑", "F1 ↑", "FAR/1000 ↓", "Event-F1 ↑"],
    ["MS-TRP", "0.6006", "0.4525", "8.12", "0.4738"],
    ["Isolation Forest", "0.4143", "0.3358", "11.43", "0.6184"],
    ["Fixed Rule", "0.4042", "0.3579", "13.61", "0.6417"],
    ["Concat-IF", "0.4008", "0.3324", "12.08", "0.6122"],
    ["Forecast Residual", "0.3941", "0.3242", "11.19", "0.6065"],
    ["Simple PPR", "0.3865", "0.3111", "10.49", "0.5930"],
    ["Graph-only", "0.3683", "0.2953", "10.25", "0.5636"],
    ["LOF", "0.2422", "0.0294", "7.47", "0.0678"],
]
P69_B = (
    "统一口径下，MS-TRP 的统一阈值 0.8059，Payment-AUPRC 0.6006、精确率 0.8593、召回率 0.3071、F1 0.4525，"
    "四项均列第一：F1 高于 Fixed Rule 的 0.3579 与 Isolation Forest 的 0.3358；"
    "FAR/1000 为 8.12，低于同档方法的 11.4 至 14.3，仅高于已退化的 LOF（7.47，召回 0.0156）。"
    "这排除了“MS-TRP 的优势只来自更保守的阈值”的解释。事件级仍然相反：统一阈值下 MS-TRP 的 Event-F1 为 0.4738，"
    "低于 Concat-IF 的 0.6122、Fixed Rule 的 0.6417 与 Isolation Forest 的 0.6184。"
    "支付级排序优势没有转化为事件覆盖优势，这与第6.7节的结论一致，也是本文不把“充分事件覆盖”写成结论的原因。"
)
P69_C = (
    "另一点需要说明：所有方法按验证集 1% 正常分位冻结后，测试期实际 FAR 普遍在每千笔 8 至 14 笔，远高于名义 0.01。"
    "2025 到 2026 的分数分布漂移对每个方法都存在，MS-TRP 的漂移最小（8.12），"
    "但这也说明单靠验证集分位阈值不能保证测试期误报预算。"
)

H610 = "6.10 跨生成机制的配对重复"
P610 = (
    "为把“换一套生成机制后优势是否还在”从单模型的绝对水平变成可比较的配对结论，"
    "在两种独立生成机制（A、B）各三个数据种子的六张库上，用同一套评估代码分别计算 MS-TRP 与四个基线的 Payment-AUPRC，"
    "按（机制，种子）配对求差 Δ = AP(MS-TRP) − AP(基线)。重复单位是数据，不是算法随机性，"
    "因此同一对内部的评测规模、时间划分与标签口径完全一致。"
    "六对差值全部为正：Isolation Forest 平均 +0.048（范围 +0.003 至 +0.145），Concat-IF +0.062（+0.021 至 +0.155），"
    "Fixed Rule +0.084（+0.029 至 +0.172），Simple PPR +0.093（+0.045 至 +0.179），每个基线的 6/6 对均为正。"
    "配对差值在机制 A 的种子 20 上最大（对 Simple PPR 达 +0.179），在机制 A 的种子 10 上对 Isolation Forest 仅 +0.003，"
    "接近持平。可见优势存在但幅度随数据抽样明显波动，六对样本也不足以给出置信区间或显著性判定。"
)

P71 = (
    "阈值口径是另一个容易混淆的因素。表4中各方法使用各自验证集冻结的阈值，表7的变体则统一在分层收缩阈值下报告；"
    "同一组分数只换阈值，实体聚集命中就从 27/43 变到 17/43（第6.3节）。因此本文把检测器比较与阈值策略比较分开报告，"
    "不把工作点差异算作检测能力差异，也不把测试期 Recall@FAR 当作部署阈值下的实际召回。"
)

P72 = (
    "现有生成机制下，共享账户子类会产生归属冲突，规则可确定性命中；共享收款人子类不产生冲突，冻结阈值下只覆盖 1/26。"
    "构造归属合法但交易结构异常的案例，以及在统一阈值下比较规则、PPR 与完整模型，是检验关系分支外部有效性的必要条件，"
    "本文尚未完成。"
)

P73_CROSS_TAIL = "逐种子的配对结果见图8，与基线的配对差见第6.10节。"

P73_TIME = (
    "时间可追溯目前只由 apply_date 一个日期键保证：模型按申请日期截断特征与图边，但数据不含字段级发生时间、可读时间、"
    "版本号与回填记录，因此不能证明历史参照、群体统计、账户归属与图拓扑都取自当时版本。"
    "要做到逐字段回放，需要在数据层补这些字段，并给出延迟补录与历史更正的完整回放案例。"
)

CONC1 = (
    "本文提出面向预算执行异常检测的 MS-TRP 模型，把多尺度稳健时序参照、异构图关系传播、可靠度门控与分层阈值校准"
    "整合为一条可追溯的风险分析流程。结果表明，支付级排序、阈值告警、事件检出必须分开评价；"
    "在当前场景里，排序性能的主要贡献来自多尺度组合。"
)
CONC2 = (
    "证据只支持部分组件。阶段边权与动态门控的差别在 0.0005 至 0.0023 之间，小于单次运行的波动量级，"
    "本文不把它们算作已验证的创新；关系相关组件的整体 AUPRC 增益有限，冻结阈值下的实体聚集召回几乎全部来自账户归属硬规则，"
    "且该规则与共享账户异常的注入方式直接对应，共享收款人子类只覆盖 1/26。"
    "阈值重冻结单独就能让实体聚集命中在 17 与 30 之间变化，说明工作点选择与检测能力必须分开报告。"
    "三个历史跨度组内模型表现都较好，事件覆盖与严格误报预算下仍有不足。"
    "未来围绕独立结构种子、跨生成机制配对重复、真实业务数据与可复现配置继续推进，才能界定这套框架的适用条件。"
)


def build_table11(doc, anchor_caption):
    ncol = len(TAB11[0])
    t = doc.add_table(rows=len(TAB11), cols=ncol)
    t.style = doc.tables[3].style
    tblPr = copy.deepcopy(doc.tables[3]._tbl.tblPr)
    old = t._tbl.tblPr
    old.getparent().replace(old, tblPr)
    widths = [Cm(2.6)] + [Cm(1.48)] * (ncol - 1)
    for ri, row in enumerate(t.rows):
        for i, cell in enumerate(row.cells):
            cell.width = widths[i]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(TAB11[ri][i])
            run.font.size = Pt(7.5)
    for cell in t.rows[0].cells:
        cell.paragraphs[0].runs[0].font.bold = True
    for col in (1, 2):  # MS-TRP 行的 AUPRC/F1
        t.rows[1].cells[col].paragraphs[0].runs[0].font.bold = True
    t.rows[3].cells[4].paragraphs[0].runs[0].font.bold = True  # Fixed Rule Event-F1
    # 列宽写入 tblGrid
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w.twips)))
    anchor_caption._p.addnext(t._tbl)
    return t


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DOCX.replace(".docx", f"_同步修订前_{stamp}.docx")
    shutil.copy2(DOCX, backup)
    print("backup:", backup)

    doc = Document(DOCX)
    P = doc.paragraphs

    # 1) 摘要
    set_text(P[2], "摘要：" + ABSTRACT)
    # 2) §4.2 传播算子
    set_text(P[55], P42_BEFORE_EQ)
    om = P[56]._p.findall(".//{%s}oMath" % M_NS)
    if om:
        om[0].addprevious(eq7_omml())
    else:
        P[56]._p.append(eq7_omml())
    for el in om:
        el.getparent().remove(el)
    set_text(P[57], P42_AFTER_EQ)
    # 3) §4.3
    set_text(P[62], P43)
    # 4) §5.1 冻结参数（索引 77 之后插入）
    perf = None
    if os.path.exists(PERF):
        perf = json.load(open(PERF, encoding="utf-8"))
    if perf:
        p51 = (
            "全部冻结参数如下，供复算核对：回跳系数 ρ=0.6（候选 0.2/0.4/0.6/0.8）、阈值收缩强度 γ=30、"
            "高分组分位点 0.01、目标误报率 0.01、门控温度 T=1.0、事件合并间隔 7 天、迭代容差 10⁻⁶ 与上限 50 轮；"
            "边权中静态关系的时间衰减为 0，支付类关系取 β=0.001/天，阶段因子 Early/Middle/Late 分别取 0.7/1.0/1.5，"
            "同质性 h=1.0；时序侧偏离饱和值 3.0、节奏饱和 0.35、年末集中饱和 0.5、Peer 条件组最少样本 5。"
            f"按当前数据重建后图含 {perf['n_nodes']:,} 个节点与 430,922 条边。"
            f"单次全量重算涵盖特征、逐评分日 as-of 传播、阈值校准与评价，在本机实测 {perf['wall_seconds']:.0f} 秒"
            f"（另一次同配置运行 191 秒），峰值内存 {perf['peak_rss_gb']:.2f} GB，"
            f"平均每次传播迭代 {perf['ppr_iters_mean']:.1f} 轮。"
            f"日终增量打分对新增一个业务日重建当日可见边的转移矩阵并跑一次 PPR，单日中位 "
            f"{perf['day_ms_median']:.0f} 毫秒、95 分位 {perf['day_ms_p95']:.0f} 毫秒、最大 "
            f"{perf['day_ms_max']:.0f} 毫秒（不含数据入库与特征重算）。"
        )
        clone_after(P[77], P[77], p51)
    # 5) §5.3 检测器/阈值分离（88 之后）
    clone_after(P[88], P[88], P52)
    # 6) §6.1 表示隔离（99 之后）
    clone_after(P[99], P[99], P61_FEATURE)
    # 7) §6.2 宏/微平均
    set_text(P[110], P62)
    # 8) §6.3 三层诊断
    set_text(P[115], P63_A)
    pa = clone_after(P[115], P[115], P63_B)
    set_text(P[116], P63_C)
    # 9) §6.4 数字
    set_text(P[127], P64)
    # 10) 新增 §6.9 / 表11 / §6.10（插在“7 讨论与局限性”之前）
    heading_tpl = P[133]     # 6.6 Heading 2
    body_tpl = P[115]
    cap_tpl = P[112]         # 表7 caption
    anchor = P[151]          # 图7 caption（其后为节7标题）
    h69 = clone_after(anchor, heading_tpl, H69)
    a1 = clone_after(h69, body_tpl, P69_A)
    cap = clone_after(a1, cap_tpl, CAP11)
    build_table11(doc, cap)
    # 表格已插到 cap 之后，后续段落要接在表后
    tbl_el = cap._p.getnext()
    # 把段落接在表格之后
    b1 = copy.deepcopy(body_tpl._p)
    tbl_el.addnext(b1)
    from docx.text.paragraph import Paragraph as _P
    para_b1 = _P(b1, cap._parent)
    set_text(para_b1, P69_B)
    b2 = clone_after(para_b1, body_tpl, P69_C)
    h610 = clone_after(b2, heading_tpl, H610)
    clone_after(h610, body_tpl, P610)
    # 11) §7.1 阈值口径（156 之后）
    clone_after(P[156], P[156], P71)
    # 12) §7.2 共享收款人限制（“眼下源头实体…”段之后，即索引 159）
    clone_after(P[159], P[159], P72)
    # 13) §7.3
    set_text(P[161], P[161].text + P73_CROSS_TAIL)
    if perf:
        p73_stats = (
            "统计证据同样有限。主结果是单次运行，0.0005 至 0.0023 的组件差异没有重复运行与配对检验；"
            "跨生成机制只有三个数据种子，报告的是最小值至最大值，不构成置信区间。"
            "支付记录存在项目、实体与事件相关，逐条独立假设会低估不确定性，后续应按项目、事件或时间块重采样。"
            "本文实测了单次全量重算的墙钟耗时（164 至 191 秒，峰值内存 0.85 GB）"
            f"与图规模（{perf['n_nodes']:,} 个节点、430,922 条边），并单独测了日终增量打分"
            f"（中位 {perf['day_ms_median']:.0f} 毫秒/日、95 分位 {perf['day_ms_p95']:.0f} 毫秒）；"
            "但增量测量只覆盖转移矩阵重建与一次 PPR，不含数据入库、特征重算与人工复核，也没有测并发与长期运行的内存曲线。"
            "窗口权重、门控温度、关系衰减、完整变体开关，都需随实现配置一起归档，才支持精确复算。"
        )
        set_text(P[166], P73_TIME)
        clone_after(P[166], P[166], p73_stats)
    # 14) 结论
    set_text(P[168], CONC1)
    set_text(P[169], CONC2)

    doc.save(DOCX)
    print("saved:", DOCX)


if __name__ == "__main__":
    main()
