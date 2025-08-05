# 🧴 Skincare Chat Bot

An AI-powered chatbot that helps users learn about skincare ingredients using a fine-tuned language model. Ask questions about skincare ingredients and get informative responses about their benefits, usage, and effects.

## Live Demo

**Try the app live on Hugging Face Spaces:**
**[skincare-chat.hf.space](https://huggingface.co/spaces/stutipandey702/skincare_chat)**

## Features

- Interactive chat interface for skincare ingredient queries
- AI-powered responses using a fine-tuned language model
- Real-time conversation with intelligent responses
- Clean, responsive web interface
- Optimized for performance and reliability

## How to Use

### Option 1: Use the Live App (Recommended)

1. **Visit the Hugging Face Space:**
   - Go to: [https://huggingface.co/spaces/stutipandey702/skincare_chat](https://huggingface.co/spaces/stutipandey702/skincare_chat)
   
2. **Wait for the app to load:**
   - The space will automatically start when you visit
   - You'll see "Building" or "Running" status at the top
   - Once ready, the chat interface will appear

3. **Start chatting:**
   - Type any question about skincare ingredients in the input box
   - Examples:
     - "What are the benefits of hyaluronic acid?"
     - "Tell me about retinol and its effects"
     - "How does niacinamide work for acne?"
   - Press Enter or click "Send"
   - Wait for the AI response

### Option 2: Run Locally

If you want to run the app on your own machine:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/stutipandey702/skincare_chat.git
   cd skincare_chat
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   - Go to: `http://localhost:7860`
   - Start chatting with the bot!

## Technical Details

### Model Architecture
- **Base Model:** GPT-2 (fallback) or TinyLlama with LoRA fine-tuning
- **Fine-tuning:** Custom LoRA adapter trained on skincare data
- **Framework:** Hugging Face Transformers with Flask backend

### Tech Stack
- **Backend:** Python, Flask, Transformers
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Hugging Face Spaces
- **Model:** Custom fine-tuned language model

### Files Structure
```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── chat.html         # Chat interface template
├── static/
│   └── style.css         # Styling for the chat interface
└── README.md             # This file
```

## Development

### Prerequisites
- Python 3.9+
- pip package manager
- Internet connection (for model downloads)

### Environment Setup
The app automatically handles model caching and environment setup. On first run, it will:
1. Download the required models
2. Set up the cache directory
3. Initialize the chat interface

### Customization
- Modify `templates/chat.html` to change the UI
- Update `static/style.css` for different styling
- Adjust model parameters in `app.py` for different response behavior

## Usage Examples

**Good prompts to try:**
- "What does vitamin C do for skin?"
- "Explain the difference between AHA and BHA"
- "Is salicylic acid good for sensitive skin?"
- "How to use peptides in skincare routine?"

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Hugging Face for providing the hosting platform and model infrastructure
- The open-source community for the underlying ML frameworks
- Kaggle dataset owner

---

**Happy skincare learning! 🧴✨**


---
title: SkincareChat
emoji: 📊
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
short_description: Skincare chat assistant to assist with ingredient questions
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
