---
title: Projects
icon: fas fa-rocket
order: 3
---

<style>
/* ===== Layout ===== */
.projects-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); /* 3 cards responsive */
  gap: 1.5rem;
  margin-top: 1.5rem;
}

/* ===== Card Design ===== */
.project-card {
  background: #1e1e1e;
  color: #fff;
  border-radius: 14px;
  padding: 1.3rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.project-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 10px 25px rgba(0,0,0,0.4);
  border-color: rgba(255, 255, 255, 0.15);
}

/* ===== Headings ===== */
.project-card h3 {
  margin-top: 0;
  font-size: 1.1rem;
  z-index: 2;
  position: relative;
}

.project-card h3 a {
  color: #f5f5f5;
  text-decoration: none !important;
  border-bottom: none !important;
}
.project-card h3 a:hover {
  color: #fff;
}

/* ===== Paragraph ===== */
.project-card p {
  color: #c7c7c7;
  margin: 0.6rem 0 1rem 0;
  z-index: 2;
  position: relative;
  line-height: 1.45;
}

/* ===== Tech Stack Badges ===== */
.tech-stack {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  z-index: 2;
  position: relative;
}

.tech-badge {
  background: rgba(255, 255, 255, 0.08);
  padding: 0.35rem 0.7rem;
  border-radius: 8px;
  font-size: 0.82rem;
  display: flex;
  align-items: center;
  gap: 6px;
  color: #e0e0e0;
  transition: background 0.3s ease;
}

.tech-badge:hover {
  background: rgba(255,255,255,0.2);
}

.tech-badge img {
  height: 18px;
}
</style>


<div class="projects-grid">

<!-- 1 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/From-Data-to-Decisions-Building-an-ANN-Model-for-Bank-Customer-Churn" target="_blank">Bank Customer Churn ANN</a></h3>
  <p>Flask app with a deep learning model that predicts if bank customers are likely to leave.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=flask" alt="">Flask</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=tensorflow" alt="">TensorFlow</div>
  </div>
</div>

<!-- 2 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/Potato-Disease-Classification-with-CNN" target="_blank">Potato Disease Classification (CNN)</a></h3>
  <p>CNN model classifying potato leaves as Early Blight, Late Blight, or Healthy using PlantVillage dataset.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=tensorflow" alt="">TensorFlow</div>
    <div class="tech-badge"><img src="https://cdn.simpleicons.org/keras" alt="Keras">Keras</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=opencv" alt="">OpenCV</div>
  </div>
</div>

<!-- 3 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/LSTM-Stock_price_prediction" target="_blank">Stock Price Prediction (LSTM)</a></h3>
  <p>LSTM network forecasting stock prices based on historical Yahoo Finance data.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
  <div class="tech-badge"><img src="https://cdn.simpleicons.org/pandas" alt="Pandas">Pandas</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=tensorflow" alt="">TensorFlow</div>
    <div class="tech-badge">Time Series</div>
  </div>
</div>

<!-- 4 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/atliq-bank-insights" target="_blank">Atliq Bank (Statistical Analysis)</a></h3>
  <p>Data cleaning, exploratory data analysis, statistical modeling, and A/B testing on synthetic banking data.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://cdn.simpleicons.org/python" alt="Python">Python</div>
    <div class="tech-badge"><img src="https://cdn.simpleicons.org/pandas" alt="Pandas">Pandas</div>
    <div class="tech-badge"><img src="https://cdn.simpleicons.org/numpy" alt="NumPy">NumPy</div>
<div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="Python">Matplotlib</div>
    <div class="tech-badge"><img src="https://cdn.simpleicons.org/scikitlearn" alt="Scikit-Learn">Scikit-Learn</div>
    <div class="tech-badge">A/B testing</div>
  </div>
</div>

<!-- 5 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/Sentiment-Analysis/tree/main" target="_blank">Sentiment Analysis Web App</a></h3>
  <p>Flask app that loads a trained NLP model to classify text as positive or negative.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=flask" alt="">Flask</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=scikitlearn" alt="">Scikit-Learn</div>
  </div>
</div>

<!-- 6 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/Email-Spam-Detector" target="_blank">Email Spam Detector</a></h3>
  <p>ML classifier using Naïve Bayes, Logistic Regression, and Random Forest with ~98% accuracy.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=scikitlearn" alt="">Scikit-Learn</div>
  </div>
</div>

<!-- 7 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/LoRA-Fine-Tune" target="_blank">LoRA Fine-Tuning (LLM)</a></h3>
  <p>Efficient LoRA fine-tuning of TinyLlama-1.1B-Chat on GSM8K dataset for reasoning tasks.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=pytorch" alt="">PyTorch</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
  <div class="tech-badge"><img src="https://cdn.simpleicons.org/huggingface" alt="Hugging Face">Hugging Face</div>
  <div class="tech-badge" style="background: rgba(255,255,255,0.08); font-weight:600; color:#e0e0e0;">
  🧩 LoRA
</div>

  <div class="tech-badge"><img src="https://cdn.simpleicons.org/jupyter" alt="Jupyter">Jupyter</div>
  </div>
</div>

<!-- 8 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/PPE-DETECTION" target="_blank">PPE Detection (YOLOv8)</a></h3>
  <p>YOLOv8 + OpenCV system to detect helmets and vests in real-time video streams.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=opencv" alt="">OpenCV</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=pytorch" alt="">PyTorch</div>
    <div class="tech-badge">
    <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/yolo.svg" 
       alt="YOLO" style="filter: invert(1); height:18px;">
  YOLO
</div>
  </div>
</div>

<!-- 9 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/RAGHub-Multi-Source-Chatbot-with-Summarization-Memory" target="_blank">RAG Chatbot</a></h3>
  <p>Streamlit chatbot using Retrieval-Augmented Generation with summarization and memory.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
  <div class="tech-badge"><img src="https://cdn.simpleicons.org/streamlit" alt="Streamlit">Streamlit</div>
  <div class="tech-badge"><img src="https://cdn.simpleicons.org/openai" alt="OpenAI">OpenAI</div>
  </div>
</div>

<!-- 10 -->
<div class="project-card">
  <h3><a href="https://github.com/chanupadeshan/LLM-Powered-Multi-Agent-Blog-Generator" target="_blank">CrewAI Content Pipeline</a></h3>
  <p>Agentic AI system with multiple specialized agents (research, writing, SEO, QA) to generate and refine blog content.</p>
  <div class="tech-stack">
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=python" alt="">Python</div>
    <div class="tech-badge"><img src="https://skillicons.dev/icons?i=pytorch" alt="">PyTorch</div>
    <div class="tech-badge" style="background: rgba(255,255,255,0.08); font-weight:600; color:#e0e0e0;">
  🤖 CrewAI
</div>

</div>



</div>

</div>
