import io
import time

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import streamlit as st
from langchain.globals import set_llm_cache
from typing import Dict, Any
from bs4 import BeautifulSoup
import PyPDF2
import requests
from urllib.parse import urlparse
from utils import dataframe_agent
from langchain.cache import InMemoryCache

# 设置缓存
set_llm_cache(InMemoryCache())

# 初始化session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_content' not in st.session_state:
    st.session_state.text_content = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'input_url' not in st.session_state:
    st.session_state.input_url = None

# 设置页面配置
st.set_page_config(
    page_title="贵中医数据分析智能体",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 美化CSS样式
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        color: #2c6e49;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        font-weight: bold;
        background: linear-gradient(to right, #f0f9eb, #e6f7ff);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #2c6e49, #4c956c);
        color: white;
        padding: 20px;
        border-radius: 0 20px 20px 0;
    }

    .sidebar .stRadio label {
        color: white !important;
        font-weight: bold;
    }

    .sidebar .stButton button {
        background-color: #fefee3;
        color: #2c6e49;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s;
    }

    .sidebar .stButton button:hover {
        background-color: #ffc9b9;
        transform: scale(1.05);
    }

    /* 卡片样式 */
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #4c956c;
    }

    /* 数据表格样式 */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }

    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(to right, #4c956c, #2c6e49);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 149, 108, 0.4);
    }

    /* 文本输入区样式 */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #d9d9d9 !important;
    }

    /* 标签样式 */
    .stRadio label {
        font-weight: bold !important;
        color: #2c6e49 !important;
    }

    /* 成功消息样式 */
    .stSuccess {
        border-radius: 10px;
        background-color: #e6f7ff !important;
    }
</style>
""", unsafe_allow_html=True)


def create_chart(input_data: Dict[str, Any], chart_type: str) -> None:
    """生成统计图表"""
    plt.style.use('seaborn-white grid')
    pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    ).set_index("x")

    if chart_type == "bar":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(input_data["columns"])))
        bars = ax.bar(input_data["columns"], input_data["data"], color=colors)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_title(input_data.get("title", "统计图表"), fontsize=14)
        ax.set_xlabel(input_data.get("x_label", "类别"), fontsize=12)
        ax.set_ylabel(input_data.get("y_label", "数值"), fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    elif chart_type == "line":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax.plot(input_data["columns"], input_data["data"], marker='o', color='#4c956c', linewidth=2.5)
        ax.set_title(input_data.get("title", "趋势图表"), fontsize=14)
        ax.set_xlabel(input_data.get("x_label", "时间/类别"), fontsize=12)
        ax.set_ylabel(input_data.get("y_label", "数值"), fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)


def process_word_file(file: io.BytesIO) -> str:
    """处理Word文档并提取文本内容"""
    import docx
    doc = docx.Document(file)
    return '\n'.join(para.text for para in doc.paragraphs)


def process_html_file(file: io.BytesIO) -> str:
    """处理HTML文件并提取文本内容"""
    soup = BeautifulSoup(file, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text()


def process_pdf_file(file: io.BytesIO) -> str:
    """处理PDF文件并提取文本内容"""
    return ''.join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)


def fetch_url_content(target_url: str) -> Dict[str, Any]:
    """从URL获取内容并解析为结构化数据"""
    try:
        if not urlparse(target_url).scheme:
            target_url = f"https://{target_url}"

        response = requests.get(target_url)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = [{
                "table_id": f"table_{i}",
                "data": pd.read_html(str(table))[0].to_dict('records'),
                "columns": list(pd.read_html(str(table))[0].columns)
            } for i, table in enumerate(soup.find_all('table'))]

            for script in soup(["script", "style"]):
                script.decompose()

            return {
                "type": "html",
                "tables": tables,
                "text": soup.get_text(strip=True),
                "status": "success"
            }
        elif 'application/pdf' in content_type:
            return {
                "type": "pdf",
                "text": ''.join(page.extract_text() for page in
                                PyPDF2.PdfReader(io.BytesIO(response.content)).pages),
                "status": "success"
            }
        return {
            "type": "unknown",
            "content": response.text,
            "status": "success"
        }
    except Exception as e :
        return {"status": "error", "message": str(e)}


def display_result(analysis_result: Dict[str, Any]) -> None:
    """根据结果类型显示输出"""
    if "answer" in analysis_result:
        with st.expander("分析结果", expanded=True):
            full_response = ""
            placeholder = st.empty()
            for chunk in analysis_result["answer"].split():
                full_response += chunk + " "
                placeholder.markdown(f"<div class='card'>{full_response}▌</div>", unsafe_allow_html=True)
                time.sleep(0.05)
            placeholder.markdown(f"<div class='card'>{full_response}</div>", unsafe_allow_html=True)

    if "table" in analysis_result:
        with st.expander("数据表格", expanded=True):
            df = pd.DataFrame(
                analysis_result["table"]["data"],
                columns=analysis_result["table"]["columns"]
            )
            st.dataframe(df.style.highlight_max(axis=0, color='#d8f3dc'))

    if "bar" in analysis_result:
        with st.expander("柱状图", expanded=True):
            create_chart(analysis_result["bar"], "bar")

    if "line" in analysis_result:
        with st.expander("折线图", expanded=True):
            create_chart(analysis_result["line"], "line")


# 侧边栏配置
with st.sidebar:
    api_vendor = st.radio(label='请选择服务提供商：',options=['OpenAI','Deepseek'])
    if api_vendor == 'OpenAI':
        base_url = 'https://twapi.openai-hk.com/v1'
        model_options = {'gpt-4o-mini','gpt-3.5-turbo','gpt-4o','gpt-4.1-mini','gpt-4.1'}
    elif api_vendor =='Deepseek':
        base_url = 'https://api.deepseek.com'
        model_options = {'deepseek-chat','deepseek-reasoner'}
    model_name = st.selectbox(label='请选择要使用的模型：',options=model_options)
    api_key = st.text_input(label='请输入你的key:',type='password')
st.header("贵中医数据分析智能体")
st.markdown("---")
st.subheader("数据输入设置")

# 数据输入方式
input_method = st.radio("数据输入方式:", ("上传文件", "输入URL"), key='input_method')

if input_method == "上传文件":
        # 文件类型选择
    file_type = st.radio(
         "文件类型:",
         ("Excel", "CSV", "Word", "HTML", "PDF"),
         key='file_type'
 )

 # 文件上传
    file_ext = {"Excel": "xlsx", "CSV": "csv",
                    "Word": "docx", "HTML": "html", "PDF": "pdf"}[file_type]
    st.session_state.uploaded_file = st.file_uploader(
            f"上传{file_type}文件",
            type=[file_ext],
            key='file_uploader'
        )
else:
        # URL输入
    st.input_url = st.text_input("输入URL:", key='input_url')

    st.markdown("---")

    # 系统信息
    st.subheader("系统信息")
    st.markdown("""
    - **版本**: 1.2.0
    - **开发者**: 贵中医AI实验室
    - **最后更新**: 2023-11-15
    """)

    st.markdown("---")
    st.markdown("### 使用说明")
    st.info("""
    1. 选择数据输入方式（上传文件或输入URL）
    2. 根据提示上传文件或输入URL
    3. 在下方输入您的数据分析需求
    4. 点击"分析"按钮获取结果
    """)



# 数据展示区域
if input_method == "上传文件" and st.session_state.uploaded_file:
    with st.container():
        st.subheader("数据预览")

        if file_type == "Excel":
            wb = openpyxl.load_workbook(st.session_state.uploaded_file)
            sheet_names = wb.sheetnames

            if len(sheet_names) > 1:
                sheet = st.selectbox("选择工作表:", sheet_names, key='sheet_name')
            else:
                sheet = sheet_names[0]

            st.session_state.df = pd.read_excel(st.session_state.uploaded_file, sheet_name=sheet)
            st.dataframe(st.session_state.df.head(2000).style.background_gradient(cmap='Blues'))

        elif file_type == "CSV":
            st.session_state.df = pd.read_csv(st.session_state.uploaded_file)
            st.dataframe(st.session_state.df.head(2000).style.background_gradient(cmap='Blues'))

        elif file_type == "Word":
            st.session_state.text_content = process_word_file(st.session_state.uploaded_file)
            with st.expander("文档内容预览", expanded=False):
                st.text_area("", value=st.session_state.text_content[:2000] + (
                    "..." if len(st.session_state.text_content) > 2000 else ""), height=300)

        elif file_type == "HTML":
            st.session_state.text_content = process_html_file(st.session_state.uploaded_file)
            with st.expander("HTML内容预览", expanded=False):
                st.text_area("", value=st.session_state.text_content[:2000] + (
                    "..." if len(st.session_state.text_content) > 2000 else ""), height=300)

        elif file_type == "PDF":
            st.session_state.text_content = process_pdf_file(st.session_state.uploaded_file)
            with st.expander("PDF内容预览", expanded=False):
                st.text_area("", value=st.session_state.text_content[:2000] + (
                    "..." if len(st.session_state.text_content) > 2000 else ""), height=300)

elif input_method == "输入URL" and st.session_state.input_url:
    with st.container():
        st.subheader("URL内容预览")
        with st.spinner("获取内容中..."):
            content = fetch_url_content(st.session_state.input_url)

            if content["status"] == "success":
                if content["type"] == "html":
                    if content["tables"]:
                        st.success(f"成功从网页中提取了 {len(content['tables'])} 个表格")
                        st.session_state.df = pd.DataFrame(content["tables"][0]["data"])
                        st.dataframe(st.session_state.df.head(10).style.background_gradient(cmap='Greens'))

                    with st.expander("网页文本内容", expanded=False):
                        st.text_area("", value=content["text"][:2000] + ("..." if len(content["text"]) > 2000 else ""),
                                     height=300)
                else:
                    st.session_state.text_content = content.get("text") or content["content"]
                    with st.expander("内容预览", expanded=False):
                        st.text_area("", value=st.session_state.text_content[:2000] + (
                            "..." if len(st.session_state.text_content) > 2000 else ""), height=300)
            else:
                st.error(f"错误: {content['message']}")

# 查询部分
st.markdown("---")
with st.container():
    st.subheader("数据分析请求")
    query = st.text_area(
        "输入您的数据分析需求:",
        height=150,
        placeholder="例如：请分析销售额最高的产品类别\n或：请展示各地区的销售趋势",
        disabled=not (st.session_state.df is not None or st.session_state.text_content is not None),
        key='query_input'
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("开始分析", key='analyze_button', use_container_width=True):
            if (input_method == "上传文件" and not st.session_state.uploaded_file) or \
                    (input_method == "输入URL" and not st.session_state.input_url):
                st.warning("请先提供数据源")
            elif not query:
                st.warning("请输入分析需求")
            else:
                with st.spinner("智能分析中，请稍候..."):
                    try:
                        if st.session_state.df is not None:
                            result = dataframe_agent(st.session_state.df, query)
                        else:
                            result = {"answer": f"文本分析结果:\n\n{st.session_state.text_content[:3000]}"}
                        display_result(result)
                    except Exception as e:
                        st.error(f"分析过程中发生错误: {str(e)}")

# 示例分析区域
st.markdown("---")
with st.expander("示例分析需求", expanded=False):
    st.markdown("""
    **针对数据表格的示例：**
    - 计算各列的平均值
    - 展示销售额前10的产品
    - 按月份统计销售趋势
    - 找出增长率最高的地区

    **针对文本内容的示例：**
    - 总结文档的主要内容
    - 提取关键信息点
    - 分析文档中的关键数据
    - 识别文档中的重点内容
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px;">
    <p>贵州中医药大学 · 实训一组 © 2023</p>
    <p>技术支持: AI实验室</p>
</div>
""", unsafe_allow_html=True)