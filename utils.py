"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.1
Date: 2025/6/25
"""
import json
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from main import model_name, base_url, api_key

PROMPT_TEMPLATE = """你是一位数据分析助手，你的回应内容取决于用户的请求内容，请按照下面的步骤处理用户请求：
1. 思考阶段 (Thought) ：先分析用户请求类型（文字回答/表格/图表），并验证数据类型是否匹配。
2. 行动阶段 (Action) ：根据分析结果选择以下严格对应的格式。
   - 纯文字回答:
     {"answer": "不超过50个字符的明确答案"}

   - 表格数据：
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}

   - 柱状图
     {"bar":{"columns": ["A", "B", "C", ...], "data":[35, 42, 29, ...]}}

   - 折线图
     {"line":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}
     
3. 格式校验要求
   - 字符串值必须使用英文双引号
   - 数值类型不得添加引号
   - 确保数组闭合无遗漏
   错误案例：{'columns':['Product', 'Sales'], data:[[A001, 200]]}
   正确案例：{"columns":["product", "sales"], "data":[["A001", 200]]}

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。

当前用户请求如下：\n"""


# 从 secrets 获取 API 密钥
#api_key = st.secrets['API_KEY']



def dataframe_agent(df, query):
    #pi_key: object = st.secrets['API_KEY']
    #load_dotenv()
    model = ChatOpenAI(
        base_url='https://api.deepseek.com',
        model="deepseek-chat",
        temperature=0,
        max_tokens=8192
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=32,
        allow_dangerous_code=True,
        verbose=True
    )

    prompt = PROMPT_TEMPLATE + query

    try:
        response = agent.invoke({"input": prompt})
        return json.loads(response["output"])
    except Exception as err:
        print(err)
        return {"answer": "暂时无法提供分析结果，请稍后重试！"}
