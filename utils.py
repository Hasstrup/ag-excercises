import os
import pandas as pd
import base64
import mimetypes

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from typing import Any

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
anthropic_client = Anthropic(api_key= anthropic_api_key) if anthropic_api_key else Anthropic()


def load_and_prepare_data(data_path):
    """
    Load and prepare data from the specified source.

    Args:
        source (str): The path to the data source (e.g., CSV file, database).
    """
    csv_path = os.path.join(os.getcwd(), data_path)
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = pd.to_datetime(df["date"], errors="coerce")
    return df

from IPython.display import HTML, display

def print_html(content: Any, title: str | None = None, is_image: bool = False):
    try:
        from html import escape as _escape
    except ImportError:
        _escape = lambda x:x
    def image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    if is_image and isinstance(content, str):
        b64 = image_to_base64(content)
        rendered = f'<img src="data:image/png;base64{b64}" alt="Image" style="max-width:100%;height:auto;border-radius:8px;">'
    elif isinstance(content, pd.DataFrame):
        rendered = content.to_html(classes="pretty-table", index=False, border=0, escape=False)
    elif isinstance(content, pd.Series):
        rendered = content.to_frame().to_html(classes="pretty-table", border=0, escape=False)
    elif isinstance(content, str):
        rendered = f"<pre><code>{_escape(content)}</code></pre>"
    else:
        rendered = f"<pre><code>{_escape(str(content))}</code></pre>"
    css = """
    <style>
    .pretty-card{
      font-family: ui-sans-serif, system-ui;
      border: 2px solid transparent;
      border-radius: 14px;
      padding: 14px 16px;
      margin: 10px 0;
      background: linear-gradient(#fff, #fff) padding-box,
                  linear-gradient(135deg, #3b82f6, #9333ea) border-box;
      color: #111;
      box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }
    .pretty-title{
      font-weight:700;
      margin-bottom:8px;
      font-size:14px;
      color:#111;
    }
    /* 🔒 Only affects INSIDE the card */
    .pretty-card pre, 
    .pretty-card code {
      background: #f3f4f6;
      color: #111;
      padding: 8px;
      border-radius: 8px;
      display: block;
      overflow-x: auto;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .pretty-card img { max-width: 100%; height: auto; border-radius: 8px; }
    .pretty-card table.pretty-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      color: #111;
    }
    .pretty-card table.pretty-table th, 
    .pretty-card table.pretty-table td {
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
    }
    .pretty-card table.pretty-table th { background: #f9fafb; font-weight: 600; }
    </style>
    """
    title_html = f'<div class="pretty-title">{title}</div>'
    card = f'<div class="pretty-card">{title_html}{rendered}</div>'
    display(HTML(css + card))

def get_response(model:str, prompt: str) -> str:
    if "claude" in model.lower() or "anthropic" in model.lower():
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": [{ "type": "text", "text": prompt }]}]
        )
        return message.content[0].text
    else:
        response = openai_client.responses.create(model=model, input=prompt)
        return response.output_text
    
def encode_image_b64(chart_path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(chart_path)
    media_type = mime or "image/png"
    with open(chart_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return media_type, encoded_string
    