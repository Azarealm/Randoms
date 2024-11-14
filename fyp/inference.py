from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

def generate_response(prompt, model, tokenizer, max_length=100):
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
        
    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating the response."

def load_model(model_path):
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def main():
    try:
        model, tokenizer = load_model("./llama_finetuned_final")
        
        print("Model loaded successfully. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break
                
            if not user_input:
                print("Please enter a message.")
                continue
                
            prompt = f"User: {user_input}\nEmotion: "
            response = generate_response(prompt, model, tokenizer)
            print(f"Assistant: {response}")
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 