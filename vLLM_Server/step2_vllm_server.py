
import time
from vllm import LLM, SamplingParams

# 加载自定义模型
llm = LLM(model="./Qwen2.5-7B-Instruct-With-Qlora")

sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.8, 
    top_p=0.9
)

# 推理
start_time = time.time()
for i in range(20):
    outputs = llm.generate(["根据下列商品描述生成一个优质的微博广告文案。商品描述：热敷毛巾面膜罩冷敷蒸脸敷脸巾美容院用面部脸部蒸汽加热面罩眼巾"]*5, sampling_params)
    # 输出结果
    for output in outputs:
        print(output.outputs[0].text)
end_time = time.time()
print((end_time - start_time) / 100)


