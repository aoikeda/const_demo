import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def validate_benchmark():
    # Initialize ChatGPT model
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0
    )
    
    # Read benchmark data
    with open('benchmark.jsonl', 'r', encoding='utf-8') as f:
        benchmark_data = [json.loads(line) for line in f]
    
    results = []
    for item in benchmark_data:
        # Create prompt to validate the answer
        prompt = f"""
以下の建築基準法に関する質問と回答のペアについて、回答の正確性を評価してください。
1から5の評価スケールで回答してください（1:完全に誤り、5:完全に正確）。
また、改善点があれば指摘してください。

質問: {item['question']}
回答: {item['answer']}

評価形式:
評価スコア: [スコア]
コメント: [評価理由と改善点]
"""
        
        # Get validation from GPT-4
        response = llm.invoke(prompt)
        
        results.append({
            'question': item['question'],
            'original_answer': item['answer'],
            'validation': response.content
        })
    
    # Save validation results
    with open('benchmark_validation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

if __name__ == "__main__":
    results = validate_benchmark()
    print("Validation completed. Results saved to benchmark_validation.json")
