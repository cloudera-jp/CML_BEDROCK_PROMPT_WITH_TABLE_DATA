import gradio as gr
import os
import json
from utils import bedrock

accept = 'application/json'
contentType = 'application/json'

with open('amp_2_app/example.txt', 'r') as file:
    example_text = file.read()
examples = {'CML Documentation': example_text}
def example_lookup(text):
  if text:
    return examples[text]
  return ''

#
# default parts of a prompt
#
system__prompt = "あなたは、関西電力の文書ファイルを、機能階層CSVで定義される階層情報によってラベル付を行う、文書ファイル管理のエキスパートです。"
example_instruction = """
以下の機能階層CSVは、関西電力の姫路第二発電所の階層情報の一部です。
1行目はヘッダで、1列目から順にグループ化され、階層関係があります。
機能階層の、発電所は第1階層、ユニットは第2階層、系統は第3階層、装置は第4階層、機器は第5階層、構成品は第6階層とも呼ばれます。
この機能階層CSVを読み込んで内容をよく理解してください。


機能階層CSV: 
発電所,ユニット,系統,装置,機器,構成品"""
functional_tier_table="""
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号循環水管,１号循環水管
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号循環水管,１号循環水管伸縮継手
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器連続除貝装置,１号復水器連続除貝装置排出ロータ＿電動機
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器連続除貝装置,１号復水器連続除貝装置
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置ボール循環ポンプ＿電動機
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置＿現地計器
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置ボール回収ストレーナ
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置ボール投入回収器
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置ボールセパレータ
新姫路第二発電所,１Ｕ,タービンおよび付属システム,１号復水装置,１号復水器細管洗浄装置,１号復水器細管洗浄装置ボール循環ポンプ
"""
example_direction="""
次に以下の文書ファイルを要約してください。
要約した内容にふさわしい機能階層CSVの値を、各階層ごとに返してください。
ふさわしいと考えられる値が複数見つかった場合は、複数の値を返してください。

文書ファイル:
"""


def clear_out():
  cleared_tuple = (gr.Textbox.update(value=""), gr.Textbox.update(value=""), gr.Textbox.update(value=""), gr.Textbox.update(value=""))
  return cleared_tuple

# List of LLM models to use for text summarization
models = ['amazon.titan-tg1-large', 'anthropic.claude-v2:1', 'anthropic.claude-3-5-sonnet-20240620-v1:0']

# Setting up the prompt syntax for the corresponding model
def prompt_construction(modelId, instruction="[instruction]", table="[table]", direction="[direction]", prompt="[input_text]"):
  if modelId == 'amazon.titan-tg1-large':
    full_prompt = instruction + """\n<text>""" + prompt + """</text>"""
  elif modelId == 'anthropic.claude-v2:1':
    full_prompt = """Human: """ + instruction + """\n<text>""" + prompt + """</text>
Assistant:"""
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    # full_prompt = [
    #   {"role": "user", "content": instruction + "\n<text>" + prompt + "</text>"}
    # ]
    full_prompt = [
      {
        "role": "user", 
        "content": f"{instruction} {table} {direction} {prompt}"
      }
    ]
  
  # print(f"Debug Prompt: {full_prompt}")

  return full_prompt

# Setting up the API call in the correct format for the corresponding model
def json_format(modelId, tokens, temperature, top_p, full_prompt="[input text]"):
  if modelId == 'amazon.titan-tg1-large':
    body = json.dumps({"inputText": full_prompt, 
                   "textGenerationConfig":{
                       "maxTokenCount":tokens,
                       "stopSequences":[],
                       "temperature":temperature,
                       "topP":top_p}})
  elif modelId == 'anthropic.claude-v2:1':
    body = json.dumps({"prompt": full_prompt,
                 "max_tokens_to_sample":tokens,
                 "temperature":temperature,
                 "top_k":250,
                 "top_p":top_p,
                 "stop_sequences":[]
                  })
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": tokens,
            "system": system__prompt,
            "messages": full_prompt,
            "temperature": temperature,
            "top_p": top_p
            }, ensure_ascii=False)
    
  print(f"Debug Body: {body}")
  
  return body

def display_format(modelId):
  if modelId == 'amazon.titan-tg1-large':
    body = json.dumps({"inputText": "[input_text]", 
                   "textGenerationConfig":{
                       "maxTokenCount":"[max_tokens]",
                       "stopSequences":[],
                       "temperature":"[temperature]",
                       "topP":"[top_p]"}})
  elif modelId == 'anthropic.claude-v2:1':
    body = json.dumps({"prompt": "[input_text]",
                 "max_tokens_to_sample":"[max_tokens]",
                 "temperature":"[temperature]",
                 "top_k":250,
                 "top_p":"[top_p]",
                 "stop_sequences":[]
                  })
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": "[max_tokens]",
            "messages": [{"role": "user", "content": "[input_text]"}],
            "temperature": "[temperature]",
            "top_p": "[top_p]"
            })
  return body

def summarize(modelId, instruction_text, custom_table, custom_direction, input_text, max_tokens, temperature, top_p):
  # Initializing the bedrock client using AWS credentials
  boto3_bedrock = bedrock.get_bedrock_client(
      region=os.environ.get("AWS_DEFAULT_REGION", None))
  
  full_prompt = prompt_construction(modelId, instruction_text, custom_table, custom_direction, input_text)
  body = json_format(modelId, max_tokens, temperature, top_p, full_prompt)

  # Foundation model is invoked here to generate a response
  response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
  response_body = json.loads(response.get('body').read())

  # Extract the output from the API response for the corresponding model
  if modelId == 'amazon.titan-tg1-large':
    result = response_body.get('results')[0].get('outputText')
  elif modelId == 'anthropic.claude-v2:1':
    result = response_body.get('completion')
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    result = response_body['content'][0]['text']

#  if isinstance(result, str):
#    return result.strip('\n')
#  elif isinstance(result, list):
#    return [item.strip('\n') for item in result if isinstance(item, str)]
#  else:
#    return str(result).strip('\n')
  
  return result.strip('\n')
#  return full_prompt

with gr.Blocks() as demo:
  with gr.Row():
    gr.Markdown("# テキスト要約からの階層情報の導出")
    example_holder = gr.Textbox(visible=False, label="サンプルテキスト", value="example")
  with gr.Row():
    modelId = gr.Dropdown(label="Bedrock Modelの選択", choices=models, value='anthropic.claude-3-5-sonnet-20240620-v1:0')
  with gr.Row():
    with gr.Column(scale=4):
      custom_instruction = gr.Textbox(label="プロンプト:", value=example_instruction)
      custom_table = gr.Textbox(label="機能階層CSV:", value=functional_tier_table)
      custom_direction = gr.Textbox(label="指示文:", value=example_direction)
      input_text = gr.Textbox(label="OCR抽出情報", placeholder="クレンジング対象のテキストを入力")
      example = gr.Examples(examples=[[example_instruction, "CML Documentation"]], inputs=[custom_instruction, example_holder])
    with gr.Column(scale=4):
      with gr.Accordion("Advanced Generation Options", open=False):
        max_new_tokens = gr.Slider(minimum=0, maximum=4096, step=1, value=512, label="Max Tokens")
        temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Temperature")
        top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=1.0, label="Top P")
      with gr.Accordion("Bedrock API Request Details", open=False):
        instruction_prompt = gr.Code(label="Instruction Prompt", value=prompt_construction('amazon.titan-tg1-large'))
        input_format = gr.JSON(label="Input Format", value=display_format('amazon.titan-tg1-large'))
        with gr.Accordion("AWS Credentials", open=False):
          label = gr.Markdown("These can be set from the project env vars")
          region = gr.Markdown("**Region**: "+os.getenv('AWS_DEFAULT_REGION'))
          access_key = gr.Markdown("**Access Key**: "+os.getenv('AWS_ACCESS_KEY_ID'))
          secret_key = gr.Markdown("**Secret Key**: *****")
      summarize_btn = gr.Button("実行", variant='primary')
      reset_btn = gr.Button("リセット")
    with gr.Column(scale=4):
      output = gr.Textbox(label="Bedrockからの応答")
  summarize_btn.click(fn=summarize, inputs=[modelId, custom_instruction, custom_table, custom_direction, input_text, max_new_tokens, temperature, top_p], outputs=output, 
                            api_name="summarize")
  reset_btn.click(fn=clear_out, inputs=[], outputs=[input_text, output, example_holder, custom_instruction], show_progress=False)
  modelId.change(fn=prompt_construction, inputs=[modelId], outputs=instruction_prompt)
  modelId.change(fn=display_format, inputs=modelId, outputs=input_format)
  example_holder.change(fn=example_lookup, inputs=example_holder, outputs=input_text, show_progress=False)

demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)
