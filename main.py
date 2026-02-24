from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import io
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from create_faiss import retrieve_context, load_retriever
from ocr import preprocess_image_for_ocr, ocr_with_preprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Test Solver")



# Initialize Hugging Face LLM
try:
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")

# Tesseract languages
tess_langs = "eng+srp"

@app.get("/")
async def root():
    return {
        "status": "API is running!",
        "model_loaded": model is not None,
        "tesseract_languages": tess_langs
    }

@app.post("/test-ocr/")
async def test_ocr(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        logger.error(f"Invalid content type: {file.content_type}")
        raise HTTPException(400, f"File must be an image, got {file.content_type}")
    
    try:
        # Read file contents
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes")
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image opened: size={image.size}, mode={image.mode}")
        
        # Perform OCR
        ocr_text = ocr_with_preprocessing(contents, lang='srp+eng')
        
        if not ocr_text:
            # Try with only Serbian if English+Serbian fails
            ocr_text = ocr_with_preprocessing(contents, lang='srp')
        
        if not ocr_text:
            return JSONResponse(
                status_code=200,
                content={"warning": "No text detected in the image.", "ocr_text": ""}
            )
        
        return {"ocr_text": ocr_text}
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing image: {str(e)}")

def answer_question_rag(index, all_chunks, embedder, model,
                        tokenizer, question, device = "cpu", max_new_tokens=30):
    # 1️⃣ Retrieve context
    context = retrieve_context(question, index=index,
                            all_chunks = all_chunks, embedder=embedder, k=3)

    # 2️⃣ Build prompt with retrieved context
    prompt = f"""
    Use the context below to answer the question clearly and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 3️⃣ Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt") # or "cpu"
    
    # 4️⃣ Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 🔥 EXTRACT ONLY THE ANSWER PART
    # Method 1: Look for "Answer:" in the response
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
    else:
        # Method 2: Remove the prompt from the response
        answer = full_response.replace(prompt, "").strip()
    
    return answer


@app.post("/solve-test/")
async def solve_test(file: UploadFile = File(...)):
    logger.info(f"Received solve-test request: {file.filename}")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # OCR
        ocr_text = pytesseract.image_to_string(image, lang=tess_langs).strip()
        if not ocr_text:
            return {"error": "No text detected in the image."}

        questions = ocr_text.split("\n")
        questions = [q.strip() for q in questions if q.strip()]

        
        # Generate answers

        answers = []

        index, all_chunks, embedder = load_retriever()
        device = "cpu"

        for q in questions:

            ans = answer_question_rag(question = q, index = index,
                                      all_chunks = all_chunks,
                                      embedder=embedder,
                                      model=model,
                                      tokenizer=tokenizer,
                                      device=device)

            answers.append(ans)

        logger.info(f"Generated text: {answers}")


        """
        
        hf_response = generator(prompt, max_length=512, do_sample=False)
        answer_text = hf_response[0]["generated_text"]
        logger.info(f"Generated text: {answer_text}")

        """
        
        # Overlay answers on image
        image_with_answers = image.copy()
        draw = ImageDraw.Draw(image_with_answers)
        
        # Try different font options
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw answers at the bottom
        draw.text((10, image.height - 200), "\n".join(answers), fill="red", font=font)
        
        # Save to bytes and return
        img_byte_arr = io.BytesIO()
        image_with_answers.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Save temporarily
        output_path = "answered_test.png"
        image_with_answers.save(output_path)
        
        return FileResponse(output_path, media_type="image/png", filename="answered_test.png")
        
    except Exception as e:
        logger.error(f"Error in solve-test: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")