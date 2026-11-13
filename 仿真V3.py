"""
预算执行仿真数据生成器 V7.0（单位年度节奏 · 年与年/月与月关联）
==============================================================
两级单位层级（主管部门→预算单位），10类图节点（指标两级拆分后），6类异常，
6种支付方式 + 公务卡报销(场景，走单位零余额) + 大额资金用款计划(支付前计划环节)。
业务口径（对齐财政真实规则）：
  · 指标两级：指标名称(如"培训费") → 明细内容/三级明细(如"教师外出培训费")，付款挂三级明细，指标是上级。
  · 资金渠道决定支付方式——财政拨款收入走零余额、教育收费收入走财政专户、
    单位自有资金/专项资金走专用存款、自筹资金走专用存款(少量实拨)。
  · 公务卡报销不是独立支付方式：报销支付走单位零余额(4步)，is_gov_card 仅为场景标记。
  · 项目类别：基本支出/经常性专项业务费/其他经常性项目/一次性/跨年度/转移支付（无"大本项目"）。

实验前强制修复（6项，见 check_data.py 验收）：
  修复1 项目生命周期：支付年份严格落在 [start_year, end_year] 内。
  修复2 收款人-账户稳定池：先生成收款人(每户1~3账户)，支付从自身账户池选，account_id 必在 bank_account。
  修复3 ENTITY_AGGREGATION 为真正图异常：Shared Payee / Shared Account，金额保持正常。
  修复4 统一事件标签结构：ground_truth_event(event_id/anomaly_type/target_*/start/end/anchor/affected/related/inject)。
  修复5 注入后不再整体缩回金额：正常数据先生成满足预算，注入后仅做格式合法性检查。
  修复6 补齐第8/9类图节点：expense_category(经济分类)、procurement_item(采购品目)。

V7.0 时间节奏（本轮新增，解决"年与年/月与月无关联"导致的 HISTORICAL_SHIFT/PROGRESS_LAG 伪异常）：
  · 每个单位一条"年度花钱节奏曲线"（12 个月占比，和=1），由年底突击程度 theta 与 12 月结算加成决定。
  · 年与年：单位曲线 = 基线曲线 + 缓慢漂移(drift) + 每年小扰动(σ=0.08)，今年节奏沿袭去年。
  · 月与月：每个（明细×年份）的年度预算固定，按节奏曲线拆成 12 个月，花多了后面就少（预算约束联动）。
  · 明细继承单位曲线再叠加自身固定扰动(δ~N(0,0.35))，形成"个体历史节奏"这一可对照基准。
  · 新增表 unit_rhythm（单位节奏留痕）、indicator_snapshot（指标月末计划/实际进度）。

输出结构（三层分离，防止标签泄漏）：
  business_data:       模型可用的业务数据（支付/项目/指标/单位/收款人/账户/凭证/审批/经济分类/采购品目）
  ground_truth_event:  仿真注入的真实异常事件标签（仅用于评价）
  audit_log:           预警/更正/退款（事后日志，不进入预测图）

使用方法：
    python 仿真V3.py              # 默认种子2026
    python 仿真V3.py --seed 42    # 指定种子
"""

import pandas as pd
import numpy as np
import sqlite3, os, sys

# ═══════════════════════════════════════════════════════════
# 0. 全局配置
# ═══════════════════════════════════════════════════════════
CFG = {
    "seed": 2026,
    "years": [2023, 2024, 2025, 2026],
    "total_payments_target": 88000,
    "max_projects_per_unit": 8,
    "max_indicators_per_project": 8,
    "output_dir": "./data_fixed",
    "db_name": "budget_data_v5.db",

    # 业务阈值（可配置，不固化地区口径）
    "large_pay_threshold_wan": 100.0,       # 区级大额支付阈值(万元)
    "large_project_threshold_wan": 3000.0,  # 区级大额项目阈值(万元)
    "auto_rewrite_type_threshold_wan": 500.0,  # 公务卡/工资超此改直付

    # 异常注入（生成器参数，与检测器分离）
    "anomaly_rate_target": (0.05, 0.08),   # 点级异常率目标(逐笔错: 单笔金额+频次突增)
    "anomaly_config": {
        "AMOUNT_SPIKE":       {"ratio": 0.050, "multipliers": [3,4,5,8,10,15]},
        "FREQ_BURST":         {"indicator_ratio": 0.04, "spike_range": (5,10)},
        "PROGRESS_LAG":       {"indicator_ratio": 0.03, "lag_ratio": 0.85},
        "YEAREND_CONC":       {"unit_ratio": 0.30, "move_ratio": 0.45},
        "HISTORICAL_SHIFT":   {"indicator_ratio": 0.03, "factors": [0.35,0.45,1.55,1.65]},
        "ENTITY_AGGREGATION": {"shared_payee_count": 40, "shared_account_count": 25, "window_days": 5},
    },
}

# ═══════════════════════════════════════════════════════════
# 1. 单位体系：两级层级（主管部门 → 预算单位）
# ═══════════════════════════════════════════════════════════

# 一级：主管部门（行政机关，parent=区本级）
DEPARTMENTS = [
    "区教育局","区卫健委","区住建委","区交通委","区水务局",
    "区农业农村委","区文旅局","区体育局","区民政局","区人社局",
    "区城管执法局","区绿化市容局","区房管局","区财政局","区审计局",
    "区司法局","区生态环境局","区应急管理局","区市场监管局",
]

# 一级：街镇（行政机关，parent=青浦区街镇）
STREETS = [
    "赵巷镇","徐泾镇","华新镇","重固镇","白鹤镇","朱家角镇",
    "练塘镇","金泽镇","夏阳街道","盈浦街道","香花桥街道",
]

# 二级：各主管部门下属的事业单位（加量，平均每个部门5-8个）
SUBORDINATES = {
    "区教育局":     ["区实验小学","区第一中学","区第二中学","区第三中学","区第四中学",
                     "区教师进修学校","区教育事务中心","区青少年活动中心","区考试中心","区社区学院"],
    "区卫健委":     ["区中心医院","区中医院","区卫生事务中心","区疾控中心","区妇幼保健所",
                     "区精神卫生中心","区急救中心","区血液管理中心"],
    "区住建委":     ["区建筑管理所","区住房保障中心","区市政管理所","区建设工程质量监督站",
                     "区房屋征收中心","区物业管理中心"],
    "区交通委":     ["区公路管理所","区交通执法大队","区港航管理中心","区城市交通运输管理所",
                     "区交通建设管理中心"],
    "区水务局":     ["区水利管理所","区排水管理所","区水文站","区水务执法大队","区供水管理中心"],
    "区农业农村委": ["区农技推广中心","区畜牧兽医站","区水产技术推广站","区农村经济管理站"],
    "区文旅局":     ["区文化馆","区图书馆","区博物馆","区旅游服务中心","区文物保护所"],
    "区体育局":     ["区体育中心","区少体校","区体育场馆管理中心","区游泳馆"],
    "区民政局":     ["区救助站","区社会福利中心","区婚姻登记中心","区社区服务中心",
                     "区养老服务中心","区殡葬管理所"],
    "区人社局":     ["区就业促进中心","区社保中心","区人才服务中心","区劳动保障监察大队",
                     "区劳动人事争议仲裁院","区职业培训中心"],
    "区城管执法局": ["区城管执法大队","区城市管理指挥中心","区市容环境管理中心"],
    "区绿化市容局": ["区市容管理所","区绿化管理所","区公园管理中心","区垃圾处理中心"],
    "区房管局":     ["区物业管理中心","区房屋征收中心","区住房保障事务中心","区房屋权属调查中心"],
    "区司法局":     ["区法律援助中心","区公证处","区社区矫正中心","区人民调解中心"],
    "区市场监管局": ["区计量检测所","区特种设备检验所","区食品药品检验所","区消费者权益保护中心"],
    "区财政局":     ["区财政事务中心","区政府采购中心","区国库收付中心"],
    "区审计局":     ["区审计事务中心","区政府投资审计中心"],
    "区生态环境局": ["区环境监测站","区环境执法大队","区环境信息中心"],
    "区应急管理局": ["区应急指挥中心","区安全生产监察大队","区防灾减灾中心"],
}

ALL_UNITS = []  # [(name, type, parent, level)]
for d in DEPARTMENTS:
    ALL_UNITS.append((d, "ADMIN", "区本级", 1))  # 主管部门都是行政机关
for s in STREETS:
    ALL_UNITS.append((s, "ADMIN", "青浦区街镇", 1))  # 街镇是行政机关
# 下属：大部分是事业单位，少数（执法大队、指挥中心等）是行政机关
SUB_IS_ADMIN = {"区城管执法大队","区交通执法大队","区水务执法大队","区环境执法大队",
                "区安全生产监察大队","区劳动保障监察大队","区城市管理指挥中心",
                "区应急指挥中心","区社区矫正中心","区国库收付中心"}
for dept, subs in SUBORDINATES.items():
    for sub in subs:
        utype = "ADMIN" if sub in SUB_IS_ADMIN else "INSTITUTION"
        ALL_UNITS.append((sub, utype, dept, 2))
TOTAL_UNITS = len(ALL_UNITS)
UNIT_ID_MAP = {}  # name -> id, filled in generate_units()

# ═══════════════════════════════════════════════════════════
# 2. 标准化标签（英文编码，不复制真实系统名称）
# ═══════════════════════════════════════════════════════════

# 4 种支付方式（一体化真实口径）：
#   财政授权支付(单位零余额) / 财政直接支付(财政零余额) / 单位资金支付(专用存款) / 实拨。
# 工资统发=代发渠道(非支付方式，工资走财政直接支付+统发清算专户收款人)；
# 财政专户拨款=教育收费预算支出的特殊操作(审核后在专用存款账户形成无指标额度再支付)，也不再单列支付方式。
PAYMENT_TYPES = [
    {"code": "UNIT_ZERO_BALANCE",  "label": "财政授权支付", "levels": 4},
    {"code": "FINANCE_ZERO_BALANCE","label": "财政直接支付", "levels": 8},
    {"code": "SPECIAL_DEPOSIT",    "label": "单位资金支付", "levels": 2},
    {"code": "DIRECT_PAYMENT",     "label": "实拨",         "levels": 8},
]
PAYMENT_TYPE_MAP = {pt["code"]: pt for pt in PAYMENT_TYPES}

# 资金渠道（财政业务口径：按资金来源划分，决定可用支付方式）
FUND_CHANNELS = [
    {"code": "FISCAL",       "label": "财政拨款收入", "ratio": 0.45},
    {"code": "EDU_FEE",      "label": "教育收费收入", "ratio": 0.10},
    {"code": "OWN_FUND",     "label": "单位自有资金", "ratio": 0.20},
    {"code": "SPECIAL_FUND", "label": "专项资金",     "ratio": 0.15},
    {"code": "SELF_RAISED",  "label": "自筹资金",     "ratio": 0.10},
]

# 资金渠道 → 可用支付方式（权重）。杜绝"财政的钱走专用存款、单位自己的钱走零余额"这类跨渠道错配。
# 教育收费收入走【财政专户拨款】→审核后在单位专用存款账户形成无指标额度，最终按"单位资金支付(专用存款)"支付。
# 工资类支付不在此表采样（工资统发：固定走财政直接支付+统发清算专户，见 generate_payments）。
FUND_CHANNEL_PAY_TYPES = {
    "FISCAL":       [("UNIT_ZERO_BALANCE", 0.55), ("FINANCE_ZERO_BALANCE", 0.45)],
    "EDU_FEE":      [("SPECIAL_DEPOSIT", 1.00)],
    "OWN_FUND":     [("SPECIAL_DEPOSIT", 1.00)],
    "SPECIAL_FUND": [("SPECIAL_DEPOSIT", 0.70), ("DIRECT_PAYMENT", 0.30)],
    # 自筹资金(未纳入部门预算)走【自筹资金采购】，支付主要是专用存款(单位自己账户的钱)；
    # 实拨支付是"备用金/活动经费/特定专项资金"这类要先拨到单位账户的钱，只保留少量。
    "SELF_RAISED":  [("SPECIAL_DEPOSIT", 0.85), ("DIRECT_PAYMENT", 0.15)],
}

ANOMALY_TYPES = [
    "AMOUNT_SPIKE",          # 单笔金额异常
    "FREQ_BURST",            # 支付频次突增
    "PROGRESS_LAG",          # 执行进度滞后
    "YEAREND_CONC",          # 年末集中支付
    "HISTORICAL_SHIFT",      # 历史节奏偏离
    "ENTITY_AGGREGATION",    # 关联实体异常聚集
]

TYPE_LEVEL_MAP = {pt["code"]: pt["levels"] for pt in PAYMENT_TYPES}

# 支出功能分类（政府收支分类科目·真实"类款项"代码，口径对齐一体化平台）
FUNC_CLASS = [("2010301","行政运行"),("2010302","一般行政管理事务"),("2010303","机关服务"),
              ("2011102","纪检监察事务"),("2011399","其他商贸事务"),
              ("2050201","学前教育"),("2050202","小学教育"),("2050203","初中教育"),
              ("2050204","高中教育"),("2050299","其他普通教育"),
              ("2060499","其他技术研究与开发"),("2069999","其他科学技术支出"),
              ("2080505","机关事业单位基本养老保险缴费支出"),("2080506","机关事业单位职业年金缴费支出"),
              ("2081199","其他残疾人事业"),("2089901","其他社会保障和就业支出"),
              ("2100101","行政运行"),("2100201","综合医院"),("2100301","城市社区卫生机构"),
              ("2100401","疾病预防控制机构"),("2100799","其他计划生育事务"),
              ("2120101","行政运行"),("2120399","其他城乡社区公共设施"),("2120501","城乡社区环境卫生"),
              ("2130101","行政运行"),("2130199","其他农业"),("2130301","行政运行"),
              ("2140101","行政运行"),("2140199","其他公路水路运输"),
              ("2150501","行政运行"),("2150599","其他工业和信息产业监管"),
              ("2210201","住房公积金"),("2210203","购房补贴")]

# 部门经济科目（真实代码）+ 所属项目分类标签：
#   PERSONNEL 人员类(工资福利/对个人补助)  OPERATION 运转类(商品服务/公用)  SPECIAL 特定目标类(资本性/专项)
ECON_CLASS = [
    ("30101","基本工资","PERSONNEL"),("30102","津贴补贴","PERSONNEL"),("30103","奖金","PERSONNEL"),
    ("30107","绩效工资","PERSONNEL"),("30113","住房公积金","PERSONNEL"),("30199","其他工资福利支出","PERSONNEL"),
    ("30302","退休费","PERSONNEL"),
    ("30201","办公费","OPERATION"),("30202","印刷费","OPERATION"),("30205","水费","OPERATION"),
    ("30206","电费","OPERATION"),("30207","邮电费","OPERATION"),("30209","物业管理费","OPERATION"),
    ("30211","差旅费","OPERATION"),("30213","维修（护）费","OPERATION"),("30215","会议费","OPERATION"),
    ("30216","培训费","OPERATION"),("30218","专用材料费","OPERATION"),("30226","劳务费","OPERATION"),
    ("30227","委托业务费","OPERATION"),("30228","工会经费","OPERATION"),("30231","公务用车运行维护费","OPERATION"),
    ("30239","其他交通费用","OPERATION"),("30299","其他商品和服务支出","OPERATION"),
    ("31002","办公设备购置","OPERATION"),("31007","信息网络及软件购置更新","OPERATION"),
    ("31001","房屋建筑物购建","SPECIAL"),("31003","专用设备购置","SPECIAL"),("31006","大型修缮","SPECIAL"),
    ("31099","其他资本性支出","SPECIAL"),
]
# 工资类科目（人员类 → 工资统发渠道）
ECON_SALARY_CODES = {c for c, _n, _cat in ECON_CLASS if c.startswith("301") or c.startswith("303")}
# 项目分类 → 可选经济科目（人员/运转按类取；特定目标类 = 资本性专项 + 委托/劳务/专用材料/维修 等）
ECON_BY_CATEGORY = {
    "PERSONNEL":      [e for e in ECON_CLASS if e[2] == "PERSONNEL"],
    "OPERATION":      [e for e in ECON_CLASS if e[2] == "OPERATION"],
    "SPECIAL_TARGET": [e for e in ECON_CLASS if e[2] == "SPECIAL" or e[0] in ("30226","30227","30218","30213")],
}

# 项目分类（一体化真实口径，两维分离）：
#   项目分类 category = 人员类 / 运转类 / 特定目标类
#   项目性质 nature   = 经常性专项业务费 / 其他经常性项目 / 专项资金 / 一次性项目
# 跨年度不是分类/性质取值，而是"经常性项目分年度支出计划"属性(默认终止年 9999)；
# 转移支付是独立模块，不在部门预算项目里，故本仿真不生成。
PROJECT_CATEGORIES = [
    {"code": "PERSONNEL",      "label": "人员类",     "ratio": 0.30},
    {"code": "OPERATION",      "label": "运转类",     "ratio": 0.40},
    {"code": "SPECIAL_TARGET", "label": "特定目标类", "ratio": 0.30},
]
PROJECT_NATURES = [
    {"code": "RECURRING_SPECIAL", "label": "经常性专项业务费", "ratio": 0.30},
    {"code": "OTHER_RECURRING",   "label": "其他经常性项目",   "ratio": 0.30},
    {"code": "SPECIAL_FUND",      "label": "专项资金",         "ratio": 0.25},
    {"code": "ONE_TIME",          "label": "一次性项目",       "ratio": 0.15},
]
# 项目分类 → 项目性质（联合分布：人员/运转类以经常性为主，特定目标类以专项/一次性为主）
CATEGORY_NATURE = {
    "PERSONNEL":      [("OTHER_RECURRING", 0.70), ("RECURRING_SPECIAL", 0.30)],
    "OPERATION":      [("OTHER_RECURRING", 0.50), ("RECURRING_SPECIAL", 0.50)],
    "SPECIAL_TARGET": [("SPECIAL_FUND", 0.40), ("ONE_TIME", 0.30), ("RECURRING_SPECIAL", 0.30)],
}
# 项目分类 → 金额规模(万元)基线
CATEGORY_SCALE = {"PERSONNEL": 800, "OPERATION": 300, "SPECIAL_TARGET": 900}
# 经常性性质(默认终止年 9999，跨年) / 非经常性(单年)
RECURRING_NATURES = {"RECURRING_SPECIAL", "OTHER_RECURRING"}

# 明细内容/三级明细 名称模板（指标名称 = 经济分类科目名；明细是更细的一层，如"培训费"→"教师外出培训费"）
# 键 = ECON_CLASS 中的部门经济科目名（与真实 301/302/303/310 类科目对齐）
DETAIL_TEMPLATES = {
    # 人员类（工资福利 / 对个人和家庭补助）—— 工资统发批次明细
    "基本工资": ["岗位工资", "薪级工资", "职务工资", "级别工资"],
    "津贴补贴": ["地区津贴", "岗位津贴", "取暖补贴", "交通补贴"],
    "奖金": ["年终一次性奖金", "绩效考核奖", "精神文明奖"],
    "绩效工资": ["基础性绩效工资", "奖励性绩效工资", "年度绩效奖"],
    "住房公积金": ["单位缴存住房公积金", "补充住房公积金"],
    "其他工资福利支出": ["探亲路费", "独生子女费", "其他工资福利"],
    "退休费": ["退休人员基本退休费", "退休人员补贴", "离休人员经费"],
    # 运转类（商品和服务支出）
    "办公费": ["日常办公用品采购", "办公耗材购置", "办公设备维护", "会议室及场地租赁"],
    "印刷费": ["文件印刷费", "宣传资料印制", "档案材料打印装订"],
    "水费": ["办公用水费", "绿化用水费"],
    "电费": ["办公用电费", "公共区域电费"],
    "邮电费": ["邮寄费", "通信费", "网络宽带费"],
    "物业管理费": ["办公楼物业管理费", "保洁服务费", "安保服务费"],
    "差旅费": ["市内交通费", "公务出差住宿费", "出差伙食补助", "城市间交通费"],
    "维修（护）费": ["办公设备维修", "房屋零星维修", "车辆维修保养"],
    "会议费": ["会议场地租赁费", "会议材料费", "参会人员住宿费"],
    "培训费": ["教师外出培训费", "干部培训班费", "职工继续教育培训费", "业务骨干培训费"],
    "专用材料费": ["专用材料采购费", "实验材料费", "试剂耗材费"],
    "劳务费": ["专家评审劳务费", "临时聘用人员劳务费", "讲课费"],
    "委托业务费": ["委托第三方审计费", "委托检测费", "外包服务费"],
    "工会经费": ["工会活动经费", "职工福利费", "工会慰问金"],
    "公务用车运行维护费": ["公务用车燃油费", "公务用车保险费", "公务用车维修费", "车辆租赁费"],
    "其他交通费用": ["租车费用", "公共交通费", "公务交通补贴"],
    "其他商品和服务支出": ["其他商品支出", "其他服务支出", "零星服务费"],
    "办公设备购置": ["办公电脑购置", "打印复印设备购置", "办公家具购置"],
    "信息网络及软件购置更新": ["软件系统开发", "信息系统运维", "硬件设备采购", "网络安全设备购置"],
    # 特定目标类（资本性支出）
    "房屋建筑物购建": ["办公楼建设", "业务用房购建", "附属设施建设"],
    "专用设备购置": ["专用仪器设备购置", "医疗设备购置", "教学设备购置"],
    "大型修缮": ["房屋大型修缮工程", "基础设施改造工程", "场地改造工程"],
    "其他资本性支出": ["无形资产购置", "其他资本性支出"],
}

# 公务卡典型报销科目（财政拨款的这些科目小额支付可用公务卡，报销走单位零余额4步）
GOV_CARD_ECON = {"办公费", "差旅费", "邮电费", "公务用车运行维护费", "印刷费", "培训费"}
PROCUREMENT = ["A0201-计算机","A0202-打印设备","C0503-车辆维修","C0503-车辆加油","C1504-保险",
               "A0206-复印机","A0206-电器","C0301-网络服务","C1901-技术测试"]
BANKS = ["工商银行","建设银行","农业银行","中国银行","交通银行",
         "浦发银行","上海银行","农商银行","招商银行","兴业银行"]

APPROVAL_ROLES = {
    1:"经办(签名)",2:"审核(签章)",3:"部门一审",4:"部门二审",
    5:"业务处一审",6:"业务处二审",7:"国库处/收付中心",8:"收付中心(开单/签章/发送)",
}

# ═══════════════════════════════════════════════════════════
# 3. 实体生成器
# ═══════════════════════════════════════════════════════════

def generate_units(seed):
    np.random.seed(seed)
    units = []
    dept_counter = 0
    street_counter = 0
    # First pass: generate IDs for L1 units (departments + streets)
    dept_ids = {}   # dept_name -> dept_id
    for name, utype, parent, level in ALL_UNITS:
        if level == 1:
            if parent == "青浦区街镇":
                street_counter += 1
                uid = f"S{str(street_counter).zfill(3)}"
                pid = "ROOT"
            else:
                dept_counter += 1
                uid = f"D{str(dept_counter).zfill(3)}"
                pid = "ROOT"
                dept_ids[name] = uid
            UNIT_ID_MAP[name] = uid
    # Second pass: generate all units with correct IDs
    sub_counters = {d: 0 for d in dept_ids}
    for name, utype, parent, level in ALL_UNITS:
        if level == 2:
            dept_id = dept_ids.get(parent, "D000")
            sub_counters[parent] += 1
            uid = f"{dept_id}{str(sub_counters[parent]).zfill(3)}"
            pid = dept_id
        else:
            uid = UNIT_ID_MAP[name]
            pid = "ROOT" if parent in ["区本级","青浦区街镇"] else ""
        UNIT_ID_MAP[name] = uid
        if parent == "青浦区街镇":
            budget = round(np.random.gamma(shape=3, scale=5000), 1)
            staff = np.random.randint(30, 200)
        elif utype == "ADMIN":
            budget = round(np.random.gamma(shape=3, scale=3000), 1)
            staff = np.random.randint(30, 200)
        else:
            budget = round(np.random.gamma(shape=2.5, scale=800), 1)
            staff = np.random.randint(10, 80)
        units.append({
            "unit_id": uid, "unit_name": name, "unit_type": "行政机关" if utype=="ADMIN" else "事业单位",
            "parent_unit_id": pid, "unit_level": level,
            "annual_budget_wan": budget, "staff_count": staff,
            "supervisor": parent,
            "salary_bank": np.random.choice(BANKS[:5]),
            "gov_card_bank": np.random.choice(BANKS[:5]),
        })
    return pd.DataFrame(units)

def generate_projects(units_df, seed):
    np.random.seed(seed)
    projects, pid = [], 0
    cat_ratios = [c["ratio"] for c in PROJECT_CATEGORIES]
    cat_map = {c["code"]: c for c in PROJECT_CATEGORIES}
    nat_map = {n["code"]: n for n in PROJECT_NATURES}
    name_pool = {
        "PERSONNEL":      ["人员经费项目", "人员经费项目", "在职人员经费"],
        "OPERATION":      ["公用经费项目", "日常运转经费", "运行保障经费"],
        "SPECIAL_TARGET": ["机关事务管理保障经费", "专项业务经费", "惠民服务项目", "基础设施建设项目",
                           "信息化建设项目", "环境整治项目", "能力提升项目", "课题研究项目", "设备更新项目"],
    }
    for _, unit in units_df.iterrows():
        n_proj = min(np.random.randint(3, CFG["max_projects_per_unit"]),
                     max(3, int(unit["annual_budget_wan"]/200)))
        for _ in range(n_proj):
            pid += 1
            cat = cat_map[np.random.choice(list(cat_map), p=cat_ratios)]
            # 项目性质：按分类的联合分布采样
            nat_pairs = CATEGORY_NATURE[cat["code"]]
            nat = nat_map[np.random.choice([c for c, _w in nat_pairs],
                                           p=np.array([w for _c, w in nat_pairs], dtype=float)/sum(w for _c, w in nat_pairs))]
            is_proc = np.random.random() < (0.15 if cat["code"] == "PERSONNEL" else 0.35)
            amt = round(np.random.gamma(shape=2.5, scale=CATEGORY_SCALE[cat["code"]]), 1)
            amt = max(amt, 50)
            big_plan = "Y" if amt >= CFG["large_project_threshold_wan"] else "N"
            # 资金渠道：人员类必为财政拨款收入（人员经费不来自自有/自筹资金）
            if cat["code"] == "PERSONNEL":
                ch = FUND_CHANNELS[0]  # FISCAL 财政拨款收入
            else:
                ch = FUND_CHANNELS[np.random.choice(len(FUND_CHANNELS), p=[c["ratio"] for c in FUND_CHANNELS])]
            # 生命周期：经常性性质 → 跨整个仿真期(终止年 9999，is_multi_year=Y)；一次性/专项 → 单年
            if nat["code"] in RECURRING_NATURES:
                sy = int(CFG["years"][0]); ey = 9999; is_multi = "Y"
            else:
                sy = int(np.random.choice(CFG["years"])); ey = sy; is_multi = "N"
            projects.append({
                "project_id": f"P{str(pid).zfill(6)}",
                "project_name": f"{unit['unit_name']}-{np.random.choice(name_pool[cat['code']])}",
                "unit_id": unit["unit_id"], "unit_name": unit["unit_name"],
                "project_category": cat["code"], "project_category_label": cat["label"],
                "project_nature": nat["code"], "project_nature_label": nat["label"],
                "total_amount_wan": amt,
                "is_procurement": "Y" if is_proc else "N",
                "is_multi_year": is_multi,
                "start_year": sy, "end_year": ey,
                "fund_channel_code": ch["code"], "fund_channel": ch["label"],
                "need_large_plan": big_plan,
                "status": np.random.choice(["RESERVED","APPROVED","EXECUTING","FINISHED"],p=[0.10,0.15,0.65,0.10]),
            })
    return pd.DataFrame(projects)

def generate_indicators(projects_df, seed):
    np.random.seed(seed)
    indicators, iid = [], 0
    for _, proj in projects_df.iterrows():
        n_ind = np.random.randint(2, min(CFG["max_indicators_per_project"], max(3, int(proj["total_amount_wan"]/80))))
        remaining = proj["total_amount_wan"]
        for j in range(n_ind):
            iid += 1
            func = FUNC_CLASS[np.random.randint(0, len(FUNC_CLASS))]
            econ_pool = ECON_BY_CATEGORY[proj["project_category"]]
            eco = econ_pool[np.random.randint(0, len(econ_pool))]
            if j == n_ind - 1:
                ind_amt = round(remaining, 1)
            else:
                p = np.random.beta(1.5, 3.0)
                ind_amt = round(remaining * min(p, 0.6), 1)
                remaining -= ind_amt
            ind_amt = max(ind_amt, 1.0)
            is_proc_ind = "Y" if (proj["is_procurement"]=="Y" and np.random.random()<0.7) else "N"
            proc_cat = np.random.choice(PROCUREMENT) if is_proc_ind=="Y" else None
            indicators.append({
                "indicator_id": f"I{str(iid).zfill(8)}",
                "project_id": proj["project_id"], "project_name": proj["project_name"],
                "unit_id": proj["unit_id"], "unit_name": proj["unit_name"],
                "indicator_name": eco[1],   # 指标名称（= 经济分类科目名，如"培训费"）
                "func_class_code": func[0], "func_class_name": func[1],
                "econ_class_code": eco[0], "econ_class_name": eco[1],
                "ind_amount_wan": ind_amt, "paid_amount_wan": 0.0,
                "is_procurement": is_proc_ind, "procurement_cat": proc_cat,
                "fund_channel_code": proj["fund_channel_code"], "fund_channel": proj["fund_channel"],
                "is_cross_dept": "Y" if np.random.random()<0.2 else "N",
                "security_audited": "Y" if np.random.random()<0.9 else "N",
            })
    return pd.DataFrame(indicators)

def generate_indicator_details(indicators_df, seed):
    """指标两级：指标(指标名称) → 明细内容/三级明细。付款挂到明细(detail_id)，指标是上级。
    明细名称来自 DETAIL_TEMPLATES（按经济分类模板），明细金额分摊指标金额。
    """
    np.random.seed(seed)
    details, did = [], 0
    for _, ind in indicators_df.iterrows():
        eco_name = ind["econ_class_name"]
        tmpl = DETAIL_TEMPLATES.get(eco_name, [f"{eco_name}明细"])
        if ind["econ_class_code"] in ECON_SALARY_CODES:
            n_det = 1  # 人员类：工资统发合并为一条明细（月度固定发放）
        else:
            n_det = np.random.randint(2, min(6, max(3, int(ind["ind_amount_wan"]/15))))
            n_det = max(2, n_det)
        remaining = ind["ind_amount_wan"]
        for j in range(n_det):
            did += 1
            if j == n_det - 1:
                det_amt = round(remaining, 1)
            else:
                p = np.random.beta(1.5, 3.0)
                det_amt = round(remaining * min(p, 0.6), 1)
                remaining -= det_amt
            det_amt = max(det_amt, 1.0)
            details.append({
                "detail_id": f"DT{str(did).zfill(8)}",
                "indicator_id": ind["indicator_id"], "indicator_name": ind["indicator_name"],
                "project_id": ind["project_id"], "project_name": ind["project_name"],
                "unit_id": ind["unit_id"], "unit_name": ind["unit_name"],
                "detail_name": np.random.choice(tmpl),
                "func_class_code": ind["func_class_code"], "func_class_name": ind["func_class_name"],
                "econ_class_code": ind["econ_class_code"], "econ_class_name": ind["econ_class_name"],
                "detail_amount_wan": det_amt, "paid_amount_wan": 0.0,
                "is_procurement": ind["is_procurement"], "procurement_cat": ind["procurement_cat"],
                "fund_channel_code": ind["fund_channel_code"], "fund_channel": ind["fund_channel"],
            })
    return pd.DataFrame(details)

def generate_expense_categories(seed):
    """修复6：第8类图节点 —— 支出经济分类(expense_category)，由经济分类科目去重形成。"""
    np.random.seed(seed)
    return pd.DataFrame([
        {"expense_cat_code": c, "expense_cat_name": n} for c, n, _cat in ECON_CLASS
    ])

def generate_procurement_items(seed):
    """修复6：第9类图节点 —— 采购品目(procurement_item)，由采购品目去重形成。"""
    np.random.seed(seed)
    rows = []
    for i, full in enumerate(PROCUREMENT, start=1):
        code, name = full.split("-", 1)
        rows.append({
            "procurement_item_id": f"PR{str(i).zfill(4)}",
            "procurement_cat": full,
            "procurement_code": code,
            "procurement_name": name,
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════
# 3.8 单位年度节奏曲线（V7.0：年与年/月与月关联）
# ═══════════════════════════════════════════════════════════

def generate_unit_rhythms(units_df, seed):
    """为每个单位生成一条"年度花钱节奏曲线"，并让曲线跨年延续。

    节奏曲线 p_u^Y = normalize( (m/12)^theta_Y · 12月加成 · exp(年扰动) )：
      · theta  —— 年底突击程度（越大越集中在年底，真实财政典型为正）；
      · drift —— 该单位固定的逐年漂移率（缓慢趋势，如"越来越年底突击"）；
      · 年扰动 —— 每月独立 N(0,0.08)，让今年节奏 ≈ 去年 + 小幅噪声。

    返回 (profiles_by_unit, rhythm_df)：
      profiles_by_unit[unit_id][year] = 12维月占比(和=1)
      rhythm_df 为可落盘的 unit_rhythm 表（留痕可复现）。
    """
    np.random.seed(seed)
    years = list(CFG["years"])
    profiles = {}
    rows = []
    for _, u in units_df.iterrows():
        uid = u["unit_id"]
        # V7.3：theta/dec_bump 降档——原 2.0/0.8 使正常曲线 68% 落在 Q4、前9月仅 32%，
        # "前9月欠执行、差额后置Q4"相对正常曲线只挪 ~0.26，PROGRESS_LAG 无偏离空间。
        # 降到 theta~1.0、dec_bump≤0.25 后，正常前9月累计 ~0.5，压缩后 ~0.1，偏离 ~0.4 可检。
        theta = float(np.clip(np.random.normal(1.0, 0.6), 0.1, 2.2))   # 年底突击程度（降档）
        dec_bump = float(np.random.uniform(0.0, 0.25))                 # 12月结算加成（收窄）
        drift = float(np.clip(np.random.normal(0.0, 0.04), -0.10, 0.10))  # 逐年漂移率
        prof = {}
        for y in years:
            theta_y = theta * (1.0 + drift * (y - years[0]))
            shape = np.array([(m / 12.0) ** theta_y for m in range(1, 13)], dtype=float)
            shape[11] *= (1.0 + dec_bump)
            shape = shape * np.exp(np.random.normal(0.0, 0.08, size=12))   # 每年小扰动
            shape = np.clip(shape, 0.0, None)
            shape = shape / shape.sum()
            prof[y] = shape
            rows.append({
                "unit_id": uid, "unit_name": u["unit_name"], "year": y,
                "theta": round(theta_y, 4), "drift": round(drift, 4),
                "december_bump": round(dec_bump, 4),
                "profile": ",".join(f"{x:.4f}" for x in shape),
            })
        profiles[uid] = prof
    return profiles, pd.DataFrame(rows)


def generate_detail_rhythm_deltas(details_df, seed):
    """每个明细一条固定的 12 维扰动 δ（N(0,0.35)），跨年不变。

    明细节奏 = 单位曲线 × δ，故同一明细"自己的历史节奏"跨年稳定（相对单位的偏差持久），
    这是 HISTORICAL_SHIFT / PROGRESS_LAG 能被"偏离往年同期"定义的前提。
    """
    np.random.seed(seed)
    delta = {}
    for _, d in details_df.iterrows():
        delta[d["detail_id"]] = np.exp(np.random.normal(0.0, 0.35, size=12))
    return delta


def generate_indicator_snapshots(indicators_df, payments_df, unit_profiles, seed):
    """生成 indicator_snapshot 月末快照表（指导书 §3.2 硬要求）。

    每个（指标×年份×月末）一行：
      due_progress    = 单位节奏曲线的累计占比（计划进度，事前已知，随年累计到1）；
      actual_progress = 该指标当年实际支付的累计金额占比（实际执行，事后累计到1）。
    PROGRESS_LAG 注入后，早期月份 actual_progress 会明显 < due_progress（欠执行后置）。
    无支付月仍保留 due_progress（计划）行，actual_progress 记 0。
    """
    np.random.seed(seed)
    rows, sid = [], 0
    pay = payments_df.copy()
    pay["year"] = pay["apply_date"].astype(str).str[:4].astype(int)
    pay["month"] = pay["apply_date"].astype(str).str[5:7].astype(int)
    for _, ind in indicators_df.iterrows():
        iid = ind["indicator_id"]
        uid = ind["unit_id"]
        up = unit_profiles.get(uid)
        sub = pay[pay["indicator_id"] == iid]
        for y in CFG["years"]:
            if up is None or y not in up:
                continue
            due_cum = np.cumsum(up[y])
            suby = sub[sub["year"] == y]
            total_amt = float(suby["amount_wan"].sum())
            for m in range(1, 13):
                if total_amt > 0:
                    cum_amt = float(suby[suby["month"] <= m]["amount_wan"].sum())
                    actual = round(cum_amt / total_amt, 4)
                else:
                    actual = 0.0
                sid += 1
                rows.append({
                    "snapshot_id": f"SNP{sid:08d}",
                    "indicator_id": iid, "unit_id": uid,
                    "month_end": f"{y}-{m:02d}",
                    "due_progress": round(float(due_cum[m - 1]), 4),
                    "actual_progress": actual,
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 4. 支付申请生成器
# ═══════════════════════════════════════════════════════════


def _days_in_month(year, month):
    """某年某月的天数（正确处理闰年2月）。"""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 30


def _rand_day(year, month):
    """返回 [1, 该月天数] 内的随机日（修复：不再缺失 29/30/31）。"""
    return int(np.random.randint(1, _days_in_month(int(year), int(month)) + 1))


def _rand_md(year):
    """返回 'MM-DD' 字符串（随机月 + 该月合法日）。"""
    m = int(np.random.randint(1, 13))
    return "%02d-%02d" % (m, _rand_day(year, m))


def generate_payments(details_df, projects_df, payees_df, accounts_df, unit_profiles, detail_delta, seed):
    np.random.seed(seed)
    target = CFG["total_payments_target"]
    amounts = details_df["detail_amount_wan"].values
    weights = np.clip(amounts / amounts.sum() * target, 5, 150).astype(int)
    total_alloc = weights.sum()
    if total_alloc < target:
        shortfall = target - total_alloc
        top = np.argsort(amounts)[-min(len(amounts), shortfall//5+1):]
        weights[top] += max(1, shortfall//len(top))
    pay_counts = np.clip(weights, 0, 200)
    # cold start: 5% details have 1-3 payments
    cold_mask = np.random.random(len(pay_counts)) < 0.05
    pay_counts[cold_mask] = np.random.randint(1, 4, size=cold_mask.sum())

    # ── 修复1：项目生命周期 → 支付年份必须落在 [start_year, end_year] 内 ──
    life = projects_df.set_index("project_id")[["start_year", "end_year"]].to_dict("index")
    base_year_weight = {y: w for y, w in zip(CFG["years"], [0.25, 0.25, 0.24, 0.26])}

    # ── 修复2：稳定收款人池路由（payee → 账户池 1~3 个）──
    pool_by_type = payees_df.groupby("payee_category")["payee_id"].apply(list).to_dict()
    payee_accs = accounts_df.groupby("payee_id")["account_id"].apply(list).to_dict()
    payee_name = payees_df.set_index("payee_id")["payee_name"].to_dict()
    payee_type = payees_df.set_index("payee_id")["payee_type"].to_dict()
    acc_info = accounts_df.set_index("account_id")[["bank_name", "bank_account"]].to_dict("index")
    salary_by_unit = payees_df[payees_df["payee_category"] == "SALARY_ACCT"].set_index("unit_id")["payee_id"].to_dict()
    salary_fallback = pool_by_type["SALARY_ACCT"][0]

    threshold = CFG["large_pay_threshold_wan"]
    statuses = ["DRAFT","SUBMITTED","APPROVED","PAID","SETTLED"]
    m_weights = np.array([0.04,0.04,0.05,0.06,0.07,0.07,0.08,0.09,0.10,0.11,0.14,0.15])
    m_probs = m_weights / m_weights.sum()
    s_early = np.array([0.12,0.18,0.23,0.27,0.20])
    s_late  = np.array([0.03,0.07,0.15,0.35,0.40])
    # 收款人类型与支付方式绑定：工资统发→工资账户、公务卡(场景)→持卡人；其余→供应商/个人/银行过渡户
    PAYEE_OTHER = ["VENDOR", "PERSONAL", "BANK_INTERIM"]
    PAYEE_OTHER_PROBS = np.array([0.65, 0.25, 0.10])
    records, pay_id = [], 0
    min_amt_map = {"办公费":0.05,"印刷费":0.03,"水费":0.50,"电费":0.50,"邮电费":0.10,"差旅费":0.10,
                   "维修（护）费":0.50,"培训费":0.20,"劳务费":0.10,"委托业务费":1.00,"工会经费":0.50,
                   "公务用车运行维护费":0.30,"办公设备购置":0.50,"专用设备购置":0.50,"房屋建筑物购建":10.00,
                   "大型修缮":10.00,"信息网络及软件购置更新":1.00,"物业管理费":0.50,"会议费":0.20,
                   "专用材料费":0.30,"其他交通费用":0.10,"其他商品和服务支出":0.10,"其他资本性支出":1.00,
                   "基本工资":0.10,"津贴补贴":0.10,"奖金":0.10,"绩效工资":0.10,"住房公积金":0.10,
                   "其他工资福利支出":0.10,"退休费":0.50}

    # 月度固定发放的人员类科目（工资统发）；奖金(30103)/其他工资福利(30199)仍走一次性口径
    MONTHLY_SALARY = {"30101", "30102", "30107", "30113", "30302"}

    for idx, (_, det) in enumerate(details_df.iterrows()):
        det_amt = det["detail_amount_wan"]
        unit_name = det["unit_name"]
        eco_name = det["econ_class_name"]
        min_a = min_amt_map.get(eco_name, 0.05)
        is_salary = det["econ_class_code"] in ECON_SALARY_CODES
        is_monthly_salary = det["econ_class_code"] in MONTHLY_SALARY

        # 该项目允许的支付年份（受生命周期约束）
        lp = life.get(det["project_id"])
        if lp is None:
            valid_years = list(CFG["years"])
        else:
            valid_years = [y for y in CFG["years"] if lp["start_year"] <= y <= lp["end_year"]] or [lp["start_year"]]
        # 年度金额份额：全局年份权重 + 明细噪声（保住 train/val/test 总量比例）
        vw = np.array([base_year_weight[y] for y in valid_years], dtype=float)
        vw = vw * np.exp(np.random.normal(0.0, 0.12, size=len(valid_years)))
        vw = vw / vw.sum()

        # 明细节奏曲线 = 单位曲线 × 明细固定扰动（V7.0：个体历史节奏）
        up = unit_profiles.get(det["unit_id"])
        dd = detail_delta.get(det["detail_id"], np.ones(12))
        year_prof = {}
        for y in valid_years:
            prof = (up[y] * dd) if (up is not None and y in up) else (m_probs * dd)
            prof = prof / prof.sum()
            year_prof[y] = prof

        # ── 修复(1)预算执行率：收紧到 ~92%（0.5~1.0），总支付金额=明细预算×执行率 ──
        exec_ratio = float(np.clip(np.random.normal(0.92, 0.08), 0.5, 1.0))
        total_to_pay = det_amt * exec_ratio

        # 计划每笔 (年, 月, 金额) 列表
        cell_amts = []

        if is_monthly_salary:
            # ── 修复(2)工资：月度固定发放，月度间仅 ±2% 微调，总额=total_to_pay ──
            n_months = len(valid_years) * 12
            monthly_base = total_to_pay / n_months
            monthly = monthly_base * np.random.normal(1.0, 0.02, n_months)
            monthly = np.clip(monthly, monthly_base * 0.9, monthly_base * 1.1)
            monthly = monthly / monthly.sum() * total_to_pay
            monthly = np.maximum(np.round(monthly, 1), 0.01)
            monthly[np.argmax(monthly)] += round(total_to_pay - monthly.sum(), 1)
            k = 0
            for y in valid_years:
                for m in range(1, 13):
                    cell_amts.append((y, m, float(monthly[k])))
                    k += 1
        else:
            n_pay = int(pay_counts[idx])
            n_pay = max(1, min(n_pay, max(1, int(total_to_pay / max(min_a, 0.01)))))
            # ── 修复(1)预算执行率：金额和=total_to_pay，不再按"月预算"结算导致蒸发 ──
            if n_pay == 1:
                amts = np.array([total_to_pay])
            else:
                amts = np.random.gamma(shape=2.0, scale=1.0, size=n_pay)
                amts = np.clip(amts, 0.0, None)
                amts = amts / amts.sum() * total_to_pay
                amts = np.maximum(np.round(amts, 1), 0.01)
                amts[np.argmax(amts)] += round(total_to_pay - amts.sum(), 1)
            # 把 n_pay 笔分配到 (年,月)（年度份额×月节奏），最大余数法
            cells = []
            for jy, y in enumerate(valid_years):
                for m in range(1, 13):
                    cells.append((y, m, jy, vw[jy] * year_prof[y][m - 1]))
            probs = np.array([c[3] for c in cells], dtype=float)
            probs = probs / probs.sum()
            raw_counts = probs * n_pay
            n_cell = np.floor(raw_counts).astype(int)
            rem = n_pay - int(n_cell.sum())
            if rem > 0:
                order = np.argsort(-(raw_counts - n_cell))
                for k in range(int(rem)):
                    n_cell[order[k]] += 1
            ai = 0
            for ci, (y, m, jy, _p) in enumerate(cells):
                nc = int(n_cell[ci])
                for _ in range(nc):
                    cell_amts.append((y, m, float(amts[ai])))
                    ai += 1

        for (y, m, amt) in cell_amts:
            pay_id += 1
            if amt < 0.01:
                amt = 0.01
            # 资金渠道决定支付方式：在渠道允许的支付方式内按权重采样
            ch_code = det["fund_channel_code"]
            allowed = FUND_CHANNEL_PAY_TYPES[ch_code]
            codes = [c for c, _w in allowed]
            wts = np.array([w for _c, w in allowed], dtype=float)
            pt_code = codes[np.random.choice(len(codes), p=wts / wts.sum())]

            # 工资统发：人员类科目(301/303)固定走财政直接支付(财政零余额)，收款人为工资代发账户
            if is_salary:
                pt_code = "FINANCE_ZERO_BALANCE"

            # 公务卡报销场景：财政拨款收入 + 小额 + 典型公务卡科目 → 支付走单位零余额(4步)，is_gov_card 仅作场景标记
            is_gov_card = "N"
            if ch_code == "FISCAL" and eco_name in GOV_CARD_ECON and amt <= 5.0 and np.random.random() < 0.15:
                is_gov_card = "Y"
                pt_code = "UNIT_ZERO_BALANCE"
                amt = round(min(amt, np.random.uniform(0.1, 5.0)), 2)

            pt = PAYMENT_TYPE_MAP[pt_code]
            amt = round(amt, 1)
            day = _rand_day(y, m)
            probs_s = s_early if m <= 9 else s_late
            probs_s = probs_s / probs_s.sum()
            status = statuses[np.random.choice(len(statuses), p=probs_s)]

            if is_gov_card == "Y":
                pcat = "CARD_HOLDER"
            elif is_salary:
                pcat = "SALARY_ACCT"
            else:
                pcat = PAYEE_OTHER[np.random.choice(len(PAYEE_OTHER), p=PAYEE_OTHER_PROBS)]

            # 从稳定收款人池选择，并从其账户池选一个账户（修复2）
            if pcat == "SALARY_ACCT":
                pid_ = salary_by_unit.get(det["unit_id"], salary_fallback)
            else:
                pid_ = np.random.choice(pool_by_type[pcat])
            acc_ids = payee_accs[pid_]
            account_id = np.random.choice(acc_ids)
            ai = acc_info[account_id]

            records.append({
                "apply_id": f"PA{str(pay_id).zfill(10)}",
                "unit_name": unit_name,
                "project_id": det["project_id"],
                "indicator_id": det["indicator_id"],
                "detail_id": det["detail_id"],
                "fund_channel_code": ch_code, "fund_channel": det["fund_channel"],
                "year": y,
                "pay_type_code": pt["code"], "pay_type_label": pt["label"],
                "approval_levels": pt["levels"],
                "is_gov_card": is_gov_card,
                "payee_id": pid_,
                "payee_name": payee_name[pid_],
                "payee_bank": ai["bank_name"],
                "payee_account": ai["bank_account"],
                "account_id": account_id,
                "public_private": payee_type[pid_],
                "func_class_code": det["func_class_code"],
                "func_class_name": det["func_class_name"],
                "econ_class_code": det["econ_class_code"],
                "econ_class_name": det["econ_class_name"],
                "procurement_cat": det["procurement_cat"],
                "amount_wan": amt,
                "approval_status": status,
                "biz_status": "SETTLED" if status=="SETTLED" else ("PAID" if status=="PAID" else "IN_FLIGHT"),
                "is_large": "Y" if amt>=threshold else "N",
                "apply_date": f"{y}-{m:02d}-{day:02d}",
                "is_executed": "Y" if status in ["PAID","SETTLED"] else "N",
                "is_gov_proc_pay": det["is_procurement"],
            })
    return pd.DataFrame(records)

def generate_large_fund_plans(projects_df, indicators_df, seed):
    """大额资金用款计划：支付前的计划环节，不是支付方式。
    触发条件：大额项目(≥大额项目阈值)且财政性资金(财政拨款/专项资金)。
    计划需国库处/收付中心(level 7)批复后方形成额度；无收款人、无实际支付金额流转。
    """
    np.random.seed(seed)
    records, pid = [], 0
    for _, proj in projects_df.iterrows():
        if proj["need_large_plan"] != "Y":
            continue
        # 财政性资金才需国库处批复的用款计划；单位自有/自筹/教育收费不走国库集中支付
        if proj["fund_channel_code"] not in ("FISCAL", "SPECIAL_FUND"):
            continue
        pid += 1
        plan_year = proj["start_year"]
        status = np.random.choice(["DRAFT", "SUBMITTED", "APPROVED", "EXECUTING", "FINISHED"],
                                  p=[0.10, 0.15, 0.25, 0.40, 0.10])
        records.append({
            "plan_id": f"PL{str(pid).zfill(8)}",
            "project_id": proj["project_id"],
            "unit_name": proj["unit_name"],
            "plan_amount_wan": proj["total_amount_wan"],
            "plan_year": plan_year,
            "plan_status": status,
            "approval_levels": 7,   # 国库处/收付中心批复
            "is_executed": "Y" if status in ("EXECUTING", "FINISHED") else "N",
            "apply_date": f"{plan_year}-{np.random.randint(1, 4):02d}-{np.random.randint(1, 32):02d}",
            "fund_channel_code": proj["fund_channel_code"],
            "fund_channel": proj["fund_channel"],
        })
    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════
# 5. 收款人 & 账户独立表
# ═══════════════════════════════════════════════════════════

def generate_payee_pool(units_df, seed):
    """修复2：先生成稳定收款人池，每个收款人固定 1~3 个账户；支付只从自己的账户池选择。
    收款人类别(category)：VENDOR 供应商 / PERSONAL 个人 / BANK_INTERIM 银行过渡户 /
    CARD_HOLDER 公务卡持卡人 / SALARY_ACCT 工资代发账户(每单位一个)。
    修复7：收款人名称采用真实业务形态——供应商=企业全称、个人/持卡人=真实姓名、
    银行过渡户=待报解/待结算户、工资=统发工资清算专户；账号统一 19 位。
    """
    np.random.seed(seed)
    payees, accounts = [], []
    pid = 0
    aid = 0
    acc_types = ["ZERO_BALANCE", "SPECIAL_DEPOSIT", "BASIC", "GENERAL"]

    # 供应商企业名池（行业 + 组织形式，前缀含地区，保证足够多样）
    VENDOR_CORE = ["建设工程", "市政工程", "园林绿化", "信息科技", "办公设备", "医疗器械",
                   "教学设备", "食品配送", "物业服务", "保安服务", "保洁服务", "印刷包装",
                   "文化传媒", "汽车维修", "教育培训", "咨询服务", "检测认证", "物流运输",
                   "劳务派遣", "餐饮管理", "环境工程", "建筑设计", "软件开发", "网络通信",
                   "家具制造", "安防监控", "环保科技", "广告装饰", "房屋修缮", "水利工程",
                   "档案管理", "人力资源", "财务咨询", "法律咨询", "会议展览", "旅游服务",
                   "体育设施", "图书文化", "仪器仪表", "精密机械"]
    VENDOR_BIZ = ["有限公司", "集团有限公司", "有限责任公司", "股份有限公司"]
    VENDOR_PREFIX = ["上海", "青浦", "上海青浦", "长三角"]
    SURNAMES = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴", "徐", "孙",
                "马", "朱", "胡", "郭", "何", "林", "罗", "高", "郑", "梁", "谢", "宋",
                "唐", "许", "韩", "冯", "邓", "曹"]
    GIVEN = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳", "杰", "娟",
             "涛", "明", "超", "秀英", "霞", "平", "刚", "桂英", "文", "辉", "丽",
             "毅", "雪", "晨", "宁", "欣", "琪", "建国", "建华", "志强", "海燕"]
    BANK_INTERIM_NAMES = ["待报解预算收入-国库集中支付代发工资清算户", "待结算财政专项资金过渡户",
                          "非税收入待报解户", "往来资金过渡户"]

    def vendor_name():
        return np.random.choice(VENDOR_PREFIX) + np.random.choice(VENDOR_CORE) + np.random.choice(VENDOR_BIZ)

    def person_name():
        return np.random.choice(SURNAMES) + np.random.choice(GIVEN)

    def bank_account_no():
        # 真实银行账号（对公/个人银行卡 16~19 位数字，这里统一 19 位）
        return "".join(str(np.random.randint(0, 10)) for _ in range(19))

    def credit_code():
        # 18 位统一社会信用代码：91 + 行政区划6位 + 组织机构9位 + 校验1位
        return "91310118" + "".join(str(np.random.randint(0, 10)) for _ in range(9)) + str(np.random.randint(0, 10))

    def add_payee(name, category, public, unit_id=None):
        nonlocal pid, aid
        pid += 1
        payee_id = f"PY{str(pid).zfill(6)}"
        n_acc = int(np.random.randint(1, 4))  # 1~3 个账户
        for _ in range(n_acc):
            aid += 1
            accounts.append({
                "account_id": f"ACCT{str(aid).zfill(6)}",
                "payee_id": payee_id,
                "bank_name": np.random.choice(BANKS),
                "bank_account": bank_account_no(),
                "account_type": np.random.choice(acc_types),
            })
        payees.append({
            "payee_id": payee_id, "payee_name": name, "payee_type": public,
            "payee_category": category, "unit_id": unit_id,
            "credit_code": (credit_code() if public == "PUBLIC" else None),
        })

    for _ in range(600):
        add_payee(vendor_name(), "VENDOR", "PUBLIC")
    for _ in range(350):
        add_payee(person_name(), "PERSONAL", "PRIVATE")
    for _ in range(40):
        add_payee(np.random.choice(BANK_INTERIM_NAMES), "BANK_INTERIM", "PUBLIC")
    for _ in range(500):
        add_payee(person_name(), "CARD_HOLDER", "PRIVATE")

    # 工资代发账户：每个单位一个（工资统发收款方 = 统发工资清算专户，国库集中支付统一清算）
    for _, u in units_df.iterrows():
        add_payee("青浦区国库集中支付统发工资清算专户", "SALARY_ACCT", "PUBLIC", unit_id=u["unit_id"])

    return pd.DataFrame(payees), pd.DataFrame(accounts)

# ═══════════════════════════════════════════════════════════
# 6. 支付凭证表
# ═══════════════════════════════════════════════════════════

def generate_vouchers(payments_df, seed):
    np.random.seed(seed)
    df = payments_df[payments_df["approval_status"].isin(["PAID","SETTLED"])].copy()
    if len(df)==0: return pd.DataFrame()
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        gen_m, gen_d = int(row["apply_date"][5:7]), int(row["apply_date"][8:10])
        send_m = np.random.randint(gen_m, 13)
        send_d = np.random.randint(gen_d if send_m==gen_m else 1, _days_in_month(int(row["year"]), send_m) + 1)
        records.append({
            "voucher_id": f"VCH{str(i+1).zfill(10)}",
            "apply_id": row["apply_id"],
            "voucher_amount_wan": row["amount_wan"],
            "voucher_status": np.random.choice(["GENERATED","SENT","RECEIPTED","SIGNED","SETTLED"],p=[0.05,0.05,0.10,0.15,0.65]),
            "gen_date": row["apply_date"],
            "send_date": f"{row['year']}-{send_m:02d}-{send_d:02d}" if row["approval_status"]=="SETTLED" else None,
        })
    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════
# 7. 一致性修复
# ═══════════════════════════════════════════════════════════

def sanitize_payments(df):
    threshold = CFG["large_pay_threshold_wan"]
    fixed = {"year_update":0, "biz_status":0, "type_rewrite":0, "large_flag":0}
    for i in df.index:
        ds = df.at[i, "apply_date"]
        if isinstance(ds, str) and len(ds) >= 4:
            ny = int(ds[:4])
            if df.at[i, "year"] != ny:
                df.at[i, "year"] = ny
                fixed["year_update"] += 1
        status = df.at[i, "approval_status"]
        exp = "SETTLED" if status=="SETTLED" else ("PAID" if status=="PAID" else "IN_FLIGHT")
        if df.at[i, "biz_status"] != exp:
            df.at[i, "biz_status"] = exp
            fixed["biz_status"] += 1
        df.at[i, "is_executed"] = "Y" if status in ["PAID","SETTLED"] else "N"
        amt = df.at[i, "amount_wan"]
        pcode = df.at[i, "pay_type_code"]
        ch = df.at[i, "fund_channel_code"]
        # 大额支付：财政拨款收入渠道走"财政直接支付(财政零余额)"；工资统发(财政直接)本身即正确，不重写
        if amt > threshold and ch == "FISCAL" and pcode != "FINANCE_ZERO_BALANCE":
            df.at[i, "pay_type_code"] = "FINANCE_ZERO_BALANCE"
            df.at[i, "pay_type_label"] = PAYMENT_TYPE_MAP["FINANCE_ZERO_BALANCE"]["label"]
            df.at[i, "approval_levels"] = PAYMENT_TYPE_MAP["FINANCE_ZERO_BALANCE"]["levels"]
            fixed["type_rewrite"] += 1
        df.at[i, "is_large"] = "Y" if amt >= threshold else "N"
    print(f"  → sanitize: year={fixed['year_update']} biz={fixed['biz_status']} type={fixed['type_rewrite']} large={fixed['large_flag']}")
    return df

# ═══════════════════════════════════════════════════════════
# 8. 异常注入器
# ═══════════════════════════════════════════════════════════

def inject_anomalies(payments_df, details_df, projects_df, indicators_df, unit_profiles, detail_delta, seed):
    np.random.seed(seed)
    events = []  # 每条元素一个异常事件，affected = 受影响 apply_id 列表（事后统一组装）
    df = payments_df.copy()
    acfg = CFG["anomaly_config"]
    det_budget = details_df.set_index("detail_id")["detail_amount_wan"].to_dict()
    life_map = projects_df.set_index("project_id")[["start_year", "end_year"]].to_dict("index")

    # 事件目标年份向 test(2026) 偏移：保证每类事件级异常在 test 有可评估的最小样本。
    TEST_YEAR = int(max(CFG["years"]))
    TEST_BIAS = 0.5

    def pick_year(years):
        """从候选年份选事件目标年：以 TEST_BIAS 概率选 test 年（若候选含它），否则均匀随机。"""
        ys = sorted({int(y) for y in years})
        if TEST_YEAR in ys and np.random.random() < TEST_BIAS:
            return TEST_YEAR
        return int(np.random.choice(ys))

    def proj_spans(pid, year):
        """项目生命周期是否覆盖某年（未知生命周期视为不限制）。"""
        lp = life_map.get(str(pid))
        if lp is None:
            return True
        return lp["start_year"] <= year <= lp["end_year"]

    def cap_amt(det_id, new_amt):
        budget = det_budget.get(det_id, 99999)
        return min(new_amt, max(budget*2.0, 100))

    # 位置索引（在添加新行前建立，后续 FREQ_BURST 增行后仅用于其自身事件）
    aid2idx = {str(a): i for i, a in enumerate(df["apply_id"])}
    # P0-2：受影响 ID 锁（日期修改登记表）。记录每笔支付已锁定到哪一年（=哪一 split）。
    # 后续注入（尤其 ENTITY_AGGREGATION）不得把已锁定成员跨年移出，保证事件成员永远落在同一 split。
    locked_year = {}

    # ── AMOUNT_SPIKE：单笔金额异常（点级；排除工资统发——固定重复支付不适用金额突增）──
    spike_pool = df.index[~df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n1 = max(100, int(len(df)*acfg["AMOUNT_SPIKE"]["ratio"]))
    n1 = min(n1, len(spike_pool))
    for i in np.random.choice(spike_pool, n1, replace=False):
        orig = df.at[i, "amount_wan"]
        mult = np.random.choice(acfg["AMOUNT_SPIKE"]["multipliers"])
        new_amt = cap_amt(df.at[i, "detail_id"], round(orig*mult, 2))
        df.at[i, "amount_wan"] = new_amt
        events.append({
            "anomaly_type": "AMOUNT_SPIKE",
            "anomaly_subtype": np.random.choice(["录入错误", "异常大额", "指标偏差"]),
            "risk_level": "HIGH" if mult > 5 else "MEDIUM",
            "target_entity_type": "PAYMENT",
            "target_entity_id": str(df.at[i, "apply_id"]),
            "related_entities": f"{df.at[i,'unit_name']},{df.at[i,'project_id']}",
            "inject_parameters": f"orig={orig:.2f},mult={mult}",
            "affected": [str(df.at[i, "apply_id"])],
        })

    # ── FREQ_BURST：支付频次突增（点级；排除工资统发）──
    freq_pool = details_df.index[~details_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n2 = max(15, int(len(details_df)*acfg["FREQ_BURST"]["indicator_ratio"]))
    n2 = min(n2, len(freq_pool))
    for ti in np.random.choice(freq_pool, n2, replace=False):
        dcode = details_df.iloc[ti]["detail_id"]
        iidx = df[df["detail_id"] == dcode].index.tolist()
        if len(iidx) < 8:
            continue
        # 频次突增集中在明细已有年份内（保证不破坏项目生命周期约束）
        detail_years = sorted({str(df.at[i, "apply_date"])[:4] for i in iidx})
        by = int(np.random.choice(detail_years))
        bm = np.random.randint(3, 12)
        ns = np.random.randint(*acfg["FREQ_BURST"]["spike_range"])
        spike_ids = []
        for _ in range(ns):
            new_row = df.iloc[iidx[0]].copy()
            new_row["apply_id"] = f"SPIKE{np.random.randint(1000,9999)}{np.random.randint(1000,9999)}"
            new_row["amount_wan"] = round(np.random.gamma(shape=2, scale=15), 1)
            new_row["year"] = by
            new_row["apply_date"] = f"{by}-{bm:02d}-{np.random.randint(1,4):02d}"
            df.loc[len(df)] = new_row
            spike_ids.append(str(new_row["apply_id"]))
        for aid in spike_ids:
            locked_year[aid] = by
        events.append({
            "anomaly_type": "FREQ_BURST",
            "anomaly_subtype": "短期高频",
            "risk_level": "MEDIUM",
            "target_entity_type": "INDICATOR_DETAIL",
            "target_entity_id": str(dcode),
            "related_entities": f"{new_row['unit_name']},{dcode}",
            "inject_parameters": f"detail={dcode},n={ns},days=2-3",
            "affected": spike_ids,
        })

    # ── PROGRESS_LAG：执行进度滞后（事件级；前9月欠执行、差额后置Q4，快照 actual<due）──
    # V7.1：改到指标级（整条指标的支付一起压），避免明细级~3笔/年把压缩信号量化淹没。
    # normal 直接取该指标所属单位的年度节奏曲线（指标无自身明细扰动，单位曲线和=1）。
    lag_pool = indicators_df.index[~indicators_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n3 = max(10, int(len(indicators_df)*acfg["PROGRESS_LAG"]["indicator_ratio"]))
    n3 = min(n3, len(lag_pool))
    for ti in np.random.choice(lag_pool, n3, replace=False):
        irow = indicators_df.iloc[ti]
        ipay = df[df["indicator_id"] == irow["indicator_id"]]
        if len(ipay) < 10:
            continue
        up = unit_profiles.get(irow["unit_id"])
        if up is None:
            continue
        # P0-2：同一事件成员必须落在同一 split（只重排某一年内月份，年不变）；
        # 目标年选该指标中"笔数足够(≥6)"的年份，确保压缩信号在累计占比上可见。
        year_cnt = ipay["apply_date"].astype(str).str[:4].value_counts()
        cand = [int(y) for y, c in year_cnt.items() if c >= 6 and int(y) in up]
        if not cand:
            continue
        ty = pick_year(cand)
        ipay_y = ipay[ipay["apply_date"].astype(str).str.startswith(str(ty))]
        normal = np.asarray(up[ty], dtype=float)
        normal = normal / normal.sum()
        lagged = normal.copy()
        early_mass = float(lagged[:9].sum())
        lagged[:9] *= 0.2                                   # 前9月只保留计划的20%
        moved = early_mass - float(lagged[:9].sum())
        late_w = lagged[9:] / lagged[9:].sum()              # 差额按Q4比例分摊
        lagged[9:] += moved * late_w
        lagged = lagged / lagged.sum()
        mids = []
        for pi in ipay_y.index:
            m = int(np.random.choice(range(1, 13), p=lagged))
            df.at[pi, "apply_date"] = f"{ty}-{m:02d}-{_rand_day(ty, m):02d}"
            mids.append(str(df.at[pi, "apply_id"]))
        for aid in mids:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "PROGRESS_LAG",
            "anomaly_subtype": "前9月欠执行,差额后置Q4",
            "risk_level": "HIGH" if irow["ind_amount_wan"] > 500 else "MEDIUM",
            "target_entity_type": "INDICATOR",
            "target_entity_id": str(irow["indicator_id"]),
            "related_entities": f"{irow['unit_name']},{irow['project_id']}",
            "inject_parameters": f"indicator={irow['indicator_id']},year={ty},early=20%plan",
            "affected": mids,
        })

    # ── YEAREND_CONC：年末集中支付（事件级；按“单位×年度”拆分，锁定单一 split）──
    n4 = max(12, int(len(ALL_UNITS)*acfg["YEAREND_CONC"]["unit_ratio"]))
    all_names = df["unit_name"].unique()
    for uname in np.random.choice(all_names, min(n4, len(all_names)), replace=False):
        upay = df[df["unit_name"] == uname]
        if len(upay) < 10:
            continue
        # P0-2：只把该单位某一年度内非12月支付挪到当年12月，事件成员锁定该年度。
        years = sorted({str(a)[:4] for a in upay["apply_date"]})
        if not years:
            continue
        ty = pick_year(years)
        upay_y = upay[upay["apply_date"].astype(str).str.startswith(str(ty))]
        non12 = upay_y[~upay_y["apply_date"].astype(str).str[5:7].eq("12")
                     & ~upay_y["econ_class_code"].isin(ECON_SALARY_CODES)].index
        if len(non12) == 0:
            continue
        mc = max(3, int(len(non12)*acfg["YEAREND_CONC"]["move_ratio"]))
        mc = min(mc, len(non12))
        moved_ids = []
        for mi in np.random.choice(non12, mc, replace=False):
            df.at[mi, "apply_date"] = f"{ty}-12-{np.random.randint(15, _days_in_month(ty, 12) + 1):02d}"
            moved_ids.append(str(df.at[mi, "apply_id"]))
        for aid in moved_ids:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "YEAREND_CONC",
            "anomaly_subtype": "年末突击",
            "risk_level": "MEDIUM",
            "target_entity_type": "UNIT",
            "target_entity_id": str(uname),
            "related_entities": str(uname),
            "inject_parameters": f"unit={uname},moved_to_dec={mc}",
            "affected": moved_ids,
        })

    # ── HISTORICAL_SHIFT：历史节奏偏离（事件级；相对该实体自身往年同期节奏反转）──
    # V7.0：不再用"金额×系数"（那只是金额异常），而是把该明细某一年内的支付月份
    # 重排成其正常节奏的"前后反转"，让"今年节奏 vs 往年同期节奏"产生真实偏离。
    shift_pool = details_df.index[~details_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n5 = max(10, int(len(details_df)*acfg["HISTORICAL_SHIFT"]["indicator_ratio"]))
    n5 = min(n5, len(shift_pool))
    for ti in np.random.choice(shift_pool, n5, replace=False):
        drow = details_df.iloc[ti]
        dpay = df[df["detail_id"] == drow["detail_id"]]
        if len(dpay) < 5:
            continue
        years = sorted({str(a)[:4] for a in dpay["apply_date"]})
        if not years:
            continue
        ty = int(np.random.choice(years))
        dpay_y = dpay[dpay["apply_date"].astype(str).str.startswith(str(ty))]
        if len(dpay_y) < 3:
            continue
        up = unit_profiles.get(drow["unit_id"])
        dd = detail_delta.get(drow["detail_id"], np.ones(12))
        if up is None or ty not in up:
            continue
        normal = up[ty] * dd
        normal = normal / normal.sum()
        shifted_p = normal[::-1]                     # 前后反转 → 明显偏离其历史节奏
        shifted_p = shifted_p / shifted_p.sum()
        shifted = []
        for pi in dpay_y.index:
            m = int(np.random.choice(range(1, 13), p=shifted_p))
            df.at[pi, "apply_date"] = f"{ty}-{m:02d}-{_rand_day(ty, m):02d}"
            shifted.append(str(df.at[pi, "apply_id"]))
        for aid in shifted:
            locked_year[aid] = int(ty)
        events.append({
            "anomaly_type": "HISTORICAL_SHIFT",
            "anomaly_subtype": f"{ty}年节奏反转(偏离往年同期)",
            "risk_level": "HIGH",
            "target_entity_type": "INDICATOR_DETAIL",
            "target_entity_id": str(drow["detail_id"]),
            "related_entities": f"{drow['unit_name']},{drow['project_id']}",
            "inject_parameters": f"detail={drow['detail_id']},year={ty},mode=reverse",
            "affected": shifted,
        })

    # ── ENTITY_AGGREGATION：真正的图异常（Shared Payee / Shared Account，金额保持正常）──
    ent = acfg["ENTITY_AGGREGATION"]
    window_days = int(ent["window_days"])
    # 重新建立 apply_id → 索引（含频次突增新增行）
    aid2idx = {str(a): i for i, a in enumerate(df["apply_id"])}

    # Shared Payee：同一收款人被多个不同项目短时间集中支付，金额正常
    # P0-2+重平衡：只选未锁定支付；目标年份向 test 偏移，且只选生命周期覆盖目标年的项目。
    payee_proj = df.groupby("payee_id")["project_id"].nunique()
    cand_payees = payee_proj[payee_proj >= 3].index.tolist()
    for _ in range(int(ent["shared_payee_count"])):
        if not cand_payees:
            break
        py = np.random.choice(cand_payees)
        ty = pick_year(CFG["years"])
        sub = df[df["payee_id"] == py]
        grp = sub.groupby("project_id")["apply_id"].apply(
            lambda lst: [a for a in lst if str(a) not in locked_year])
        grp = grp[grp.map(len) > 0]
        # 只保留生命周期覆盖目标年的项目，保证移动后不破坏生命周期且事件落单一 split
        grp = grp[[p for p in grp.index if proj_spans(p, ty)]]
        if len(grp) < 3:
            continue
        n_proj = min(np.random.randint(3, 5), len(grp))
        chosen = np.random.choice(grp.index.tolist(), n_proj, replace=False)
        affected = [str(np.random.choice(grp[p])) for p in chosen]
        mm = np.random.randint(2, 12)
        dd = np.random.randint(1, 29 - window_days)
        for k, ap in enumerate(affected):
            df.at[aid2idx[ap], "apply_date"] = f"{ty}-{mm:02d}-{min(dd + k, _days_in_month(ty, mm)):02d}"
        for aid in affected:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "ENTITY_AGGREGATION",
            "anomaly_subtype": "Shared Payee",
            "risk_level": "HIGH",
            "target_entity_type": "PAYEE",
            "target_entity_id": str(py),
            "related_entities": f"payee={py},projects={','.join(chosen)}",
            "inject_parameters": f"shared_payee={py},n_projects={n_proj},window={window_days}d",
            "affected": affected,
        })

    # Shared Account：不同收款人的支付短时间通过同一账户聚集，金额正常
    # P0-2+重平衡：只选未锁定支付；目标年份向 test 偏移，且只选项目覆盖目标年的支付。
    for _ in range(int(ent["shared_account_count"])):
        per_payee = df.drop_duplicates("payee_id")
        if len(per_payee) < 2:
            break
        ty = pick_year(CFG["years"])
        n = min(np.random.randint(2, 4), len(per_payee))
        picks = per_payee.sample(n)
        payees_inv = picks["payee_id"].tolist()
        affected = []
        ok = True
        for py in payees_inv:
            sub = df[df["payee_id"] == py]
            unc = sub[~sub["apply_id"].astype(str).isin(set(locked_year))]
            unc = unc[unc["project_id"].map(lambda p: proj_spans(p, ty))]
            if len(unc) == 0:
                ok = False
                break
            affected.append(str(unc.iloc[0]["apply_id"]))
        if not ok or len(affected) < 2:
            continue
        shared = df.at[aid2idx[affected[0]], "account_id"]
        shared_bank = df.at[aid2idx[affected[0]], "payee_bank"]
        shared_acc = df.at[aid2idx[affected[0]], "payee_account"]
        mm = np.random.randint(2, 12)
        dd = np.random.randint(1, 29 - window_days)
        for k, ap in enumerate(affected):
            idx = aid2idx[ap]
            df.at[idx, "account_id"] = shared
            df.at[idx, "payee_bank"] = shared_bank
            df.at[idx, "payee_account"] = shared_acc
            df.at[idx, "apply_date"] = f"{ty}-{mm:02d}-{min(dd + k, _days_in_month(ty, mm)):02d}"
        for aid in affected:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "ENTITY_AGGREGATION",
            "anomaly_subtype": "Shared Account",
            "risk_level": "HIGH",
            "target_entity_type": "ACCOUNT",
            "target_entity_id": str(shared),
            "related_entities": f"account={shared},payees={','.join(payees_inv)}",
            "inject_parameters": f"shared_account={shared},n_payees={n},window={window_days}d",
            "affected": affected,
        })

    # ── 统一事件标签结构（修复4）——
    # 修复5：注入后不再整体缩回金额，只做格式合法性检查（由 sanitize_payments 负责）。
    final_idx = {str(a): i for i, a in enumerate(df["apply_id"])}
    rows = []
    for eid, ev in enumerate(events, start=1):
        affected = [a for a in ev["affected"] if a in final_idx]
        if not affected:
            continue
        dates = sorted(str(df.at[final_idx[a], "apply_date"]) for a in affected)
        rows.append({
            "event_id": f"EV{eid:06d}",
            "anomaly_type": ev["anomaly_type"],
            "anomaly_subtype": ev.get("anomaly_subtype"),
            "risk_level": ev.get("risk_level"),
            "target_entity_type": ev.get("target_entity_type"),
            "target_entity_id": ev.get("target_entity_id"),
            "start_time": dates[0],
            "end_time": dates[-1],
            "anchor_apply_id": affected[0],
            "affected_apply_ids": ",".join(affected),
            "related_entities": ev.get("related_entities"),
            "inject_parameters": ev.get("inject_parameters"),
        })
    labels_df = pd.DataFrame(rows)
    return df, labels_df

# ═══════════════════════════════════════════════════════════
# 9. 审批/预警/更正/退款（事后日志）
# ═══════════════════════════════════════════════════════════

def generate_approvals(payments_df, seed):
    np.random.seed(seed)
    records = []
    for _, row in payments_df.iterrows():
        if row["approval_status"] == "DRAFT": continue
        lv = int(row.get("approval_levels", 0))
        if lv == 0: continue
        s = row["approval_status"]
        if s in ["PAID","SETTLED"]:
            passed, final = lv, "PASS"
        elif s == "APPROVED":
            passed, final = lv-1, "PASS"
        elif s == "SUBMITTED":
            passed, final = max(1, np.random.randint(1, lv)), "IN_FLIGHT"
        else:
            passed, final = np.random.randint(1, lv+1), "REJECT"
        for lv_i in range(1, lv+1):
            if lv_i <= passed:
                r, c = ("REJECT", np.random.choice(["金额有误","指标错误","收款人信息不全","附件不全"])) if (lv_i==passed and final=="REJECT") else ("PASS", None)
            else:
                r, c = "PENDING", None
            records.append({
                "apply_id": row["apply_id"], "unit_name": row["unit_name"],
                "role": APPROVAL_ROLES.get(lv_i, f"LEVEL{lv_i}"),
                "level": lv_i, "result": r, "reject_reason": c,
                "approval_date": f"{row['year']}-{_rand_md(row['year'])}",
            })
    return pd.DataFrame(records)

def generate_warnings(payments_df, seed):
    np.random.seed(seed)
    records = []
    for _, row in payments_df.iterrows():
        amt = row["amount_wan"]
        prob = 0.7 if amt>CFG["auto_rewrite_type_threshold_wan"] else (0.15 if amt>100 else (0.05 if amt>50 else 0.01))
        if np.random.random() < prob:
            wtype = np.random.choice(["NOTICE","FREEZE","BLOCK"], p=[0.50,0.35,0.15])
            ws = np.random.choice(["IN_FLIGHT","RELEASED","VOIDED","BANNED"], p=[0.30,0.40,0.15,0.15])
            records.append({
                "apply_id": row["apply_id"], "unit_name": row["unit_name"],
                "warning_type": wtype,
                "trigger_reason": np.random.choice(["累计执行接近上限","金额超历史均值3倍","收款人异常","分类不匹配","余额不足","疑似重复"]),
                "warning_status": ws,
                "trigger_time": row["apply_date"],
                "resolve_time": f"{row['year']}-{_rand_md(row['year'])}" if ws!="IN_FLIGHT" else None,
            })
    return pd.DataFrame(records)

def generate_corrections(payments_df, seed):
    np.random.seed(seed)
    records = []
    for _, row in payments_df.iterrows():
        if row["approval_status"] != "SETTLED": continue
        prob = 0.05 if "ZERO_BALANCE" in row["pay_type_code"] else 0.02
        if np.random.random() < prob:
            fc = np.random.random() < 0.3
            records.append({
                "apply_id": row["apply_id"], "unit_name": row["unit_name"],
                "pay_type_code": row["pay_type_code"],
                "orig_amount_wan": row["amount_wan"],
                "new_amount_wan": round(row["amount_wan"]*np.random.uniform(0.65,1.35),2),
                "subject_changed": "Y" if fc else "N",
                "correction_reason": np.random.choice(["科目调整","金额调整","收款人变更","资金来源调整"]),
                "extra_audit": "Y" if fc else "N",
                "correction_date": f"{row['year']}-{_rand_md(row['year'])}",
            })
    return pd.DataFrame(records)

def generate_refunds(payments_df, seed):
    np.random.seed(seed)
    records = []
    for _, row in payments_df.iterrows():
        if row["approval_status"] not in ["PAID","SETTLED"]: continue
        m = int(row["apply_date"][5:7]) if isinstance(row["apply_date"],str) else 6
        prob = 0.04 if m>=11 else 0.01
        if np.random.random() < prob:
            records.append({
                "apply_id": row["apply_id"], "unit_name": row["unit_name"],
                "orig_amount_wan": row["amount_wan"],
                "refund_amount_wan": round(row["amount_wan"]*np.random.uniform(0.1,1.0),2),
                "refund_reason": np.random.choice(["合同变更","重复支付","验收不合格","政策调整","银行退票","主动退款"]),
                "settle_status": np.random.choice(["SETTLED","UNSETTLED"],p=[0.7,0.3]),
                "refund_date": f"{row['year']}-{_rand_md(row['year'])}",
            })
    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════
# 10. 数据字典 & 统计
# ═══════════════════════════════════════════════════════════

def generate_data_dict(conn, output_dir):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    desc = {
        "unit":"预算单位(两级层级)", "project":"预算项目", "indicator":"预算指标(指标名称级)",
        "indicator_detail":"明细内容/三级明细(付款挂接级)",
        "large_fund_plan":"大额资金用款计划(支付前计划环节,无收款人)",
        "expense_category":"支出经济分类(第8类图节点)",
        "procurement_item":"采购品目(第9类图节点)",
        "payment_apply":"支付申请(business_data)", "payment_voucher":"支付凭证",
        "payee":"收款人", "bank_account":"银行账户",
        "payment_approval":"审批记录", "ground_truth_event":"真实异常事件标签(只用于评价)",
        "warning":"监控预警(audit_log)", "payment_correction":"支付更正(audit_log)", "refund":"退款(audit_log)",
        "unit_rhythm":"单位年度节奏留痕(12月形状+逐年漂移,V7.0)",
        "indicator_snapshot":"指标月末计划/实际进度快照(进度滞后异常依据,V7.0)",
    }
    records = []
    for (tn,) in tables:
        cursor.execute(f"PRAGMA table_info({tn})")
        for col in cursor.fetchall():
            records.append({
                "table_name": tn, "table_desc": desc.get(tn,""),
                "column_name": col[1], "data_type": col[2],
                "nullable": "N" if col[3]==1 else "Y",
                "default_value": col[4] if col[4] is not None else "",
            })
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "data_dictionary.csv"), index=False, encoding="utf-8-sig")
    print(f"  → data dict: {len(records)} fields")

def print_stats(payments_df, ground_truth, details_df, indicators_df=None, units_df=None, projects_df=None, payees_df=None):
    total = len(payments_df)

    def _affected(sub):
        ids = set()
        if "affected_apply_ids" in sub.columns:
            for v in sub["affected_apply_ids"].dropna():
                for pid in str(v).split(","):
                    pid = pid.strip()
                    if pid and pid != "N/A":
                        ids.add(pid)
        if "anchor_apply_id" in sub.columns:
            for v in sub["anchor_apply_id"].dropna():
                if v and str(v) != "N/A":
                    ids.add(str(v))
        return ids

    # 点级异常（逐笔错：单笔金额异常 + 频次突增）
    point_ids = _affected(ground_truth[ground_truth["anomaly_type"].isin(
        ["AMOUNT_SPIKE", "FREQ_BURST"])])
    # 全部受影响支付（含模式类异常的 affected_apply_ids）
    anom_ids = _affected(ground_truth)
    point_rate = len(point_ids)/total*100
    rate = len(anom_ids)/total*100
    lo, hi = CFG["anomaly_rate_target"]
    flag = "[OK]" if lo*100 <= point_rate <= hi*100 else "[WARN]"
    print(f"\n  [Stats]")
    print(f"    Payments: {total:,}  Point anomalies(逐笔错): {len(point_ids):,}  "
          f"Point rate: {point_rate:.2f}% {flag}")
    print(f"    All affected(含模式类): {len(anom_ids):,}  Rate: {rate:.2f}%")
    ndet = len(details_df)
    nind = len(indicators_df) if indicators_df is not None else 0
    pc = payments_df.groupby("detail_id").size()
    cold = (pc<=3).sum()
    print(f"    Details: {ndet:,}  Indicators: {nind:,}  Avg seq len: {total/ndet:.1f}  "
          f"Cold-start(<=3): {cold}/{ndet}({cold/ndet*100:.1f}%)")
    if units_df is not None and projects_df is not None and payees_df is not None:
        nn = len(units_df)+len(projects_df)+nind+ndet+total+len(payees_df)
        ne = total*3 + ndet + nind + len(projects_df)
        density = ne/(nn*nn) if nn>0 else 0
        print(f"    Graph nodes: {nn:,}  Edges: ~{ne:,}  Density: {density:.6f}")
    paid = payments_df.groupby("detail_id")["amount_wan"].sum()
    budget = details_df.set_index("detail_id")["detail_amount_wan"]
    over = (paid>budget).sum()
    print(f"    Over-budget details: {over}/{ndet} ({over/ndet*100:.1f}%) {'[OK]' if over/ndet<0.25 else '[WARN]'}")
    units_l1 = units_df[units_df["unit_level"]==1] if units_df is not None else None
    units_l2 = units_df[units_df["unit_level"]==2] if units_df is not None else None
    if units_l1 is not None:
        print(f"    Units: {len(units_df)} total (L1:{len(units_l1)}, L2:{len(units_l2)})")
    return point_rate

# ═══════════════════════════════════════════════════════════
# 11. 主程序
# ═══════════════════════════════════════════════════════════

def main():
    if "--seed" in sys.argv:
        CFG["seed"] = int(sys.argv[sys.argv.index("--seed")+1])
    if "--out" in sys.argv:
        CFG["output_dir"] = sys.argv[sys.argv.index("--out")+1]
    print("="*60)
    print(f"  Budget Simulation V7.0  |  seed={CFG['seed']}  |  {TOTAL_UNITS} units")
    print("="*60)

    np.random.seed(CFG["seed"])
    out = CFG["output_dir"]
    os.makedirs(out, exist_ok=True)
    db_path = os.path.join(out, CFG["db_name"])
    if os.path.exists(db_path): os.remove(db_path)

    # 1-3: Entities
    print("\n[1/8] Units...")
    units_df = generate_units(CFG["seed"])
    print("[1.1/8] Unit annual rhythms (V7.0 年与年/月与月)...")
    unit_profiles, unit_rhythm_df = generate_unit_rhythms(units_df, CFG["seed"] + 15)
    print(f"  -> {len(unit_rhythm_df)} unit-year rhythm rows")
    print(f"  → {len(units_df)} units (L1:{len(units_df[units_df['unit_level']==1])}, L2:{len(units_df[units_df['unit_level']==2])})")

    print("[2/8] Projects...")
    projects_df = generate_projects(units_df, CFG["seed"]+1)
    print(f"  → {len(projects_df)} projects (procurement:{(projects_df['is_procurement']=='Y').sum()})")

    print("[3/8] Indicators...")
    indicators_df = generate_indicators(projects_df, CFG["seed"]+2)
    print(f"  → {len(indicators_df)} indicators  budget:{indicators_df['ind_amount_wan'].sum():.0f}万")

    print("[3.2/8] Indicator details (三级明细)...")
    details_df = generate_indicator_details(indicators_df, CFG["seed"]+12)
    detail_delta = generate_detail_rhythm_deltas(details_df, CFG["seed"] + 16)
    print(f"  → {len(details_df)} details")

    print("[3.5/8] Large-fund plans (大额资金用款计划)...")
    plans_df = generate_large_fund_plans(projects_df, indicators_df, CFG["seed"]+11)
    print(f"  → {len(plans_df)} plans")

    print("[3.6/8] Expense categories & procurement items (第8/9类图节点)...")
    expense_cat_df = generate_expense_categories(CFG["seed"]+13)
    procurement_item_df = generate_procurement_items(CFG["seed"]+14)
    print(f"  → {len(expense_cat_df)} expense categories, {len(procurement_item_df)} procurement items")

    # 收款人池 & 账户（修复2：先建稳定池，支付再从池内选）
    print("[3.7/8] Payee pool & accounts...")
    payees_df, accounts_df = generate_payee_pool(units_df, CFG["seed"]+9)
    print(f"  → {len(payees_df)} payees, {len(accounts_df)} accounts")

    # 4: Payments
    print("[4/8] Payment applications...")
    payments_df = generate_payments(details_df, projects_df, payees_df, accounts_df, unit_profiles, detail_delta, CFG["seed"]+3)
    print(f"  → {len(payments_df)} payments")

    # 5: Anomalies + sanitize
    print("[5/8] Inject anomalies + sanitize...")
    payments_df, ground_truth = inject_anomalies(payments_df, details_df, projects_df, indicators_df, unit_profiles, detail_delta, CFG["seed"]+4)
    payments_df = sanitize_payments(payments_df)
    for at in ANOMALY_TYPES:
        print(f"     - {at}: {(ground_truth['anomaly_type']==at).sum()}")

    print("[5.1/8] Indicator month-end snapshots (due/actual progress)...")
    snapshots_df = generate_indicator_snapshots(indicators_df, payments_df, unit_profiles, CFG["seed"] + 17)
    print(f"  -> {len(snapshots_df)} indicator-month snapshots")

    # Update detail/indicator paid amounts
    paid_det = payments_df.groupby("detail_id")["amount_wan"].sum()
    for i, row in details_df.iterrows():
        details_df.at[i, "paid_amount_wan"] = round(paid_det.get(row["detail_id"], 0.0), 1)
    paid_ind = payments_df.groupby("indicator_id")["amount_wan"].sum()
    for i, row in indicators_df.iterrows():
        indicators_df.at[i, "paid_amount_wan"] = round(paid_ind.get(row["indicator_id"], 0.0), 1)

    # 7: Vouchers
    print("[7/8] Vouchers...")
    vouchers_df = generate_vouchers(payments_df, CFG["seed"]+10)
    print(f"  → {len(vouchers_df)} vouchers")

    # 8: Approvals, warnings, corrections, refunds
    print("[8/8] Audit logs (approvals/warnings/corrections/refunds)...")
    approvals_df = generate_approvals(payments_df, CFG["seed"]+5)
    warnings_df = generate_warnings(payments_df, CFG["seed"]+6)
    corrections_df = generate_corrections(payments_df, CFG["seed"]+7)
    refunds_df = generate_refunds(payments_df, CFG["seed"]+8)
    print(f"  → {len(approvals_df)} approvals, {len(warnings_df)} warnings, {len(corrections_df)} corrections, {len(refunds_df)} refunds")

    # Save
    print("\nSaving...")
    conn = sqlite3.connect(db_path)
    # business_data
    for df, name in [(units_df,"unit"),(projects_df,"project"),(indicators_df,"indicator"),
                      (details_df,"indicator_detail"),
                      (plans_df,"large_fund_plan"),
                      (expense_cat_df,"expense_category"),(procurement_item_df,"procurement_item"),
                      (payments_df,"payment_apply"),(vouchers_df,"payment_voucher"),
                      (payees_df,"payee"),(accounts_df,"bank_account"),(approvals_df,"payment_approval")]:
        df.to_sql(name, conn, if_exists="replace", index=False)
    # ground_truth_event（统一事件标签结构，修复4）
    ground_truth.to_sql("ground_truth_event", conn, if_exists="replace", index=False)
    # V7.0 时间节奏留痕表
    unit_rhythm_df.to_sql("unit_rhythm", conn, if_exists="replace", index=False)
    snapshots_df.to_sql("indicator_snapshot", conn, if_exists="replace", index=False)
    # audit_log
    for df, name in [(warnings_df,"warning"),(corrections_df,"payment_correction"),(refunds_df,"refund")]:
        df.to_sql(name, conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_unit ON payment_apply(unit_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_date ON payment_apply(apply_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_account ON payment_apply(account_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_type ON ground_truth_event(anomaly_type)")
    generate_data_dict(conn, out)
    # Export CSVs
    for tbl in ["unit","project","indicator","indicator_detail","large_fund_plan","expense_category","procurement_item",
                 "payment_apply","payment_voucher","payee","bank_account","payment_approval",
                 "ground_truth_event","warning","payment_correction","refund","unit_rhythm","indicator_snapshot"]:
        try:
            pd.read_sql(f"SELECT * FROM {tbl}", conn).to_csv(os.path.join(out, f"{tbl}.csv"), index=False, encoding="utf-8-sig")
        except: pass
    conn.close()

    print_stats(payments_df, ground_truth, details_df, indicators_df, units_df, projects_df, payees_df)
    print(f"\n{'='*60}")
    print(f"  V7.0 Complete | {len(payments_df):,} payments | seed={CFG['seed']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
