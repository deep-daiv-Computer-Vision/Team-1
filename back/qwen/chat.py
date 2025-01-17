import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import APIRouter, Depends
from database import MongoClient
from database.service import get_current_user

# TODO Streaming
# from starlette.responses import JSONResponse, StreamingResponse

router = APIRouter()
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct" # Qwen/Qwen2.5-Coder-3B-Instruct
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,      
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
    cache_dir='/workspace'
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir='/workspace'
)


async def parse_response(response, cot_prompt="Chain of Thought:"):
    
    # code
    code_blocks = re.findall(r"'''(?:python)?\n(.*?)'''", response, re.DOTALL)
    
    # CoT
    cot_pattern = rf"{re.escape(cot_prompt)}(.*?)(?:\n\n|\Z)"
    chain_of_thought_blocks = re.findall(cot_pattern, response, re.DOTALL)

    # message block
    without_code = re.sub(r"'''(?:python)?\n.*?'''", "", response, flags=re.DOTALL)
    without_cot = re.sub(cot_pattern, "", without_code, flags=re.DOTALL)
    message_blocks = without_cot.strip()

    return code_blocks, message_blocks, chain_of_thought_blocks


@router.post('/create-chat')
async def create_chat(chat_name: str, user=Depends(get_current_user)):
    await MongoClient.get_client().chat.users.update_one(
        {"email": user['email']},
        {"$push": {"chat": {"chat_room": chat_name, "messages": [{}]}}},
        )
    return {"chat_name": chat_name}


@router.post('chat')
async def chat(chat_name: str, prompt: str, user_message: str, user=Depends(get_current_user)):

    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user_message}
    ]

    # text tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # inference
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    
    # response, decode
    generated_ids_trimmed = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    # 설명, 코드, 추론 과정
    message, codes, chain_of_thoughts = parse_response(response)

    # DB 저장
    await MongoClient.get_client().chat.users.update_one(
    {"email": user['email'], "chat.chat_name": chat_name},
        {
            "$set": {
                "chat.$.messages": messages,
                "chat.$.codes": codes,  
                "chat.$.chain_of_thoughts": chain_of_thoughts 
            }
        }
    )

    return {"message": message, "code": codes, "chain_of_thought": chain_of_thoughts}




                                                        