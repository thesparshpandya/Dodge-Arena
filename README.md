# **Smart Dodge Arena: ML-Powered Survival**

**Smart Dodge Arena** is an adaptive, high-speed 2D arcade game built with Python and Pygame. Instead of relying on random number generators or hardcoded enemy logic, the game uses real-time **Machine Learning (scikit-learn)** to analyze your playstyle, map your spatial preferences, and predict your future movements to actively hunt you down.

---

## **Key Features**
* **Real-Time Data Pipeline:** The game continuously logs your positional data at 60 FPS without impacting performance.
* **K-Means Spatial Clustering:** Uses `sklearn.cluster.KMeans` to dynamically group your movement history into three personalized "Safe Zones" based on where you actually spend your time.
* **Naive Bayes Prediction:** Uses `sklearn.naive_bayes.GaussianNB` to analyze your transitions between zones and predict exactly where you will attempt to dodge next.
* **Adaptive Difficulty:** As your score increases, block speed and spawn rates scale aggressively.
* **Modern UI/UX:** Features a scrolling parallax grid, glassmorphism UI overlays, dynamic player movement trails, and glowing entities.

---

## **Prerequisites**
You will need **Python 3.8 or higher** installed on your system. 

## **Installation & Setup**
**1. Clone or download the repository:**
Extract the project files into a dedicated folder.

**2. Create a virtual environment (Recommended):**
```bash
python -m venv .venv

```

*(Activate it using `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux).*

**3. Install the dependencies:**

```bash
pip install -r requirements.txt

```

---

## **How to Play**

**1. Launch the game:**

```bash
python main.py

```

**2. Controls:**

* **Left Arrow:** Move Left
* **Right Arrow:** Move Right
* **Spacebar:** Reboot/Restart (on Game Over)

**3. Gameplay Loop:**

* Survive as long as possible by dodging the falling red blocks.
* Watch the **ML Diagnostics Panel** in the top left. Once it switches from "ANALYZING TARGET..." to "SYSTEM ACTIVE", the Machine Learning models have collected enough data to begin predicting your movements and targeting your preferred zones.