from flask import Flask, request, jsonify
from vllm import LLM

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型
llm = LLM(model="./Qwen2.5-7B-Instruct-With-Qlora")

@app.route("/generate", methods=["POST"])
def generate():
    """
    接收 POST 请求，生成文本。
    请求格式：
    {
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 50)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        # 检查 prompt 是否为空
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # 调用模型生成
        outputs = llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # 返回生成结果
        results = [output.text for output in outputs]
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)