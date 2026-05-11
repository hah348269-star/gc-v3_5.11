class LLMModule:
    def __init__(self):
        self.policies = [
            {"name": "北京市碳排放权交易管理办法", "score": 0.95},
            {"name": "中关村绿色建筑补贴", "score": 0.92},
            {"name": "北京市光伏补贴", "score": 0.88},
            {"name": "超低能耗建筑奖励", "score": 0.86},
            {"name": "储能设施补贴", "score": 0.84}
        ]

    def match_policy(self, query):
        return self.policies

    def generate_report(self):
        return "智能报告：预计年减排 2320 吨，投资回收期 4.2 年..."

    def chat(self, query):
        return "我可以帮你分析碳排放、匹配政策、生成报告~"