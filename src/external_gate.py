"""
Q2-3A —— 无标签外部数据时间权限门（external_time_gate）
==========================================================
外部零样本/迁移数据（如 USAspending）只有在 score_time_valid=true 时才允许评分；
score_time_valid=false 时一律返回 status=not_comparable 并禁止继续零样本推理。

当前 Q2-3A 未启动（指导书允许【待填】），本模块只落地门控契约，供
test_external_time_gate 断言；不虚构外部数据性能。
"""


def external_time_gate(score_time_valid):
    """时间权限门：false → not_comparable（禁止零样本推理）。"""
    if not bool(score_time_valid):
        return {"status": "not_comparable",
                "reason": "score_time_valid=false；禁止继续零样本推理"}
    return {"status": "allowed"}
