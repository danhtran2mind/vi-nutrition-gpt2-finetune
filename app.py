import torch
import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_nutrition_text(
    input_text,
    model_name='danhtran2mind/vi-nutrition-gpt2-finetune',
    max_length=100,
    min_length=100,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    num_beams=5,
    num_return_sequences=1,
    random_seed=42
):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate text
    sample_outputs = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        max_length=int(max_length),
        min_length=int(min_length),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        num_beams=int(num_beams),
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=int(num_return_sequences)
    )
    
    # Decode and return the generated texts
    generated_texts = [
        tokenizer.decode(output.tolist(), skip_special_tokens=True)
        for output in sample_outputs
    ]
    
    return "\n\n".join([f"Generated Output {i+1}:\n{text}" for i, text in enumerate(generated_texts)])

# Gradio interface with improved theme and grouped arguments
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Nutrition Text Generation App
        Generate nutrition-related text using a fine-tuned GPT-2 model. Enter a prompt and adjust parameters to customize the output.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_text = gr.Textbox(
                label="Prompt",
                value="Các nhóm chất dinh dưỡng cần thiết cho cơ thể hằng ngày gồm",
                placeholder="Enter a nutrition-related prompt",
                lines=3,
                info="Provide a starting sentence or phrase for the model to generate text."
            )
            
            with gr.Accordion("Text Length Settings", open=True):
                max_length = gr.Slider(
                    50, 500, value=100, step=10, label="Max Length",
                    info="Maximum number of tokens in the generated text."
                )
                min_length = gr.Slider(
                    50, 500, value=100, step=10, label="Min Length",
                    info="Minimum number of tokens in the generated text."
                )
            
            with gr.Accordion("Generation Parameters", open=False):
                temperature = gr.Slider(
                    0.1, 2.0, value=0.7, step=0.1, label="Temperature",
                    info="Controls randomness. Lower values make output more focused."
                )
                top_k = gr.Slider(
                    10, 100, value=40, step=10, label="Top K",
                    info="Limits sampling to the top K most likely tokens."
                )
                top_p = gr.Slider(
                    0.1, 1.0, value=0.9, step=0.1, label="Top P",
                    info="Limits sampling to the smallest set of tokens with cumulative probability above P."
                )
                num_beams = gr.Slider(
                    1, 10, value=5, step=1, label="Number of Beams",
                    info="Number of beams for beam search. Higher values increase quality but slow down generation."
                )
            
            with gr.Accordion("Output Settings", open=False):
                num_return_sequences = gr.Slider(
                    1, 5, value=1, step=1, label="Number of Sequences",
                    info="Number of different outputs to generate."
                )
                random_seed = gr.Slider(
                    0, 2**32, value=42, step=1, label="Random Seed",
                    info="Seed for reproducible results."
                )
            
            submit_button = gr.Button("Generate Text", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Generated Output")
            output = gr.Textbox(
                label="Result",
                lines=15,
                placeholder="Generated text will appear here...",
                show_copy_button=True
            )
    
    submit_button.click(
        fn=generate_nutrition_text,
        inputs=[
            input_text,
            gr.State(value='danhtran2mind/vi-nutrition-gpt2-finetune'),
            max_length,
            min_length,
            temperature,
            top_k,
            top_p,
            num_beams,
            num_return_sequences,
            random_seed
        ],
        outputs=output
    )

demo.launch()