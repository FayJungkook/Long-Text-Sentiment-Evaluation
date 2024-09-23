import requests


def call_model(model_name, user_input):
    if model_name == "yi-large":
        import openai
        from openai import OpenAI
        client = OpenAI(api_key="yours", base_url="https://api.lingyiwanwu.com/v1")
        completion = client.chat.completions.create(
            model="yi-large",
            messages=[{"role": "user", "content": user_input}]
        )
        return completion

    elif model_name == "qwen-long":
        import openai
        from openai import OpenAI
        client = OpenAI(api_key="yours", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        completion = client.chat.completions.create(
            model="qwen-long",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        result = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
        return result

    elif model_name == "glm-4-9B":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key="yours")
        response = client.chat.completions.create(
            model="glm-4-9B",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message
    
    elif model_name == "glm-4-long":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key="yours")
        response = client.chat.completions.create(
            model="glm-4-long",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message
    
    elif model_name == "glm-4-0520":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key="yours")
        response = client.chat.completions.create(
            model="glm-4-0520",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message

    elif model_name == "ernie-speed-128k":
        import json

        import requests
        API_KEY = "yours"
        SECRET_KEY = "sZh0Mn2pUnUnEsphobSPXRXle5a4wGLs"
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + get_access_token(API_KEY, SECRET_KEY)
        payload = json.dumps({
            "messages": [{"role": "user", "content": user_input}],
            "temperature": 0.95,
            "top_p": 0.7,
            "penalty_score": 1,
            "collapsed": True
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.text

    elif model_name == "moonshot-v1-128k":
        import openai
        from openai import OpenAI
        client = OpenAI(api_key="yours", base_url="https://api.moonshot.cn/v1")
        messages = [
            {
                "role": "system",
                "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手??",
            },
            {"role": "user", "content": user_input},
        ]
        completion = client.chat.completions.create(model="moonshot-v1-128k", messages=messages, temperature=0.3)
        return completion.choices[0].message

def get_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    return str(requests.post(url, params=params).json().get("access_token"))

