 🌸 DecodeLabs AI Engineering Internship — Week 2
Project 2: Enterprise Data Classification Using AI
Internship Track: AI Engineering Intern
Framework: Scikit-Learn | IPO Pipeline (Input → Process → Output)
Dataset: Iris Benchmark Dataset (UCI / Fisher 1936)
Algorithm: K-Nearest Neighbors (KNN) Classifier
Standard Tuning: K = 5 (Blueprint Compliance)
Edition: ⭐ VVIP ULTRA PROFESSIONAL ⭐
Batch: 2026 | Powered by DecodeLabs
🎬 Preview — Terminal Experience
plain
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DECODELABS AI INTERNSHIP                           ║
║              PROJECT 2: DATA CLASSIFICATION USING AI | VVIP ULTRA          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🤖 Algorithm: K-Nearest Neighbors (KNN)                                    ║
║  📊 Dataset: Iris Benchmark (UCI / Fisher 1936)                            ║
║  🏗️ Architecture: IPO Pipeline (Input → Process → Output)                     ║
║  🎨 Theme: CYBERPUNK | Animations: ENABLED | Interactive: ENABLED            ║
╠══════════════════════════════════════════════════════════════════════════════╣

  ⠋ Loading Iris Dataset...
  ✓ Loading Iris Dataset complete!

    ➤ Dataset: Iris Benchmark (UCI / Fisher 1936)
    ➤ Samples: 150 | Features: 4
    ➤ Classes: 3 — setosa, versicolor, virginica

  ✓ Split Complete | Train: 120 | Test: 30 | Stratified: ✓
  ✓ Gatekeeper Rule Applied → Data Leakage Prevented 🔒
    ➤ Post-Scale Mean (F0): 0.0000 (Target: 0.0)
    ➤ Post-Scale Var (F0): 1.0000 (Target: 1.0)

  ▸ Elbow Evaluation: [████████████████████████░░░░░░░░░░░░░░░░] 60.0%
  ✓ Elbow Evaluation complete!

    ➤ Optimal K (Data-Driven): K=1 | Error: 0.0333
    ➤ Blueprint K (Mandated): K=5 [STRICT COMPLIANCE]

  ⠋ Training Final Model (K=5)...
  ✓ Training Final Model (K=5) complete!

  ✓ Model Trained | K=5 | Metric=Euclidean | Weights=Uniform

  ⠋ Generating Ultra Visualizations...
  ✓ Dashboard Saved → decodelabs_knn_diagnostics_dashboard.png
  ✓ Animation Saved → decodelabs_knn_diagnostics_animated.gif
  ✓ Confusion Matrix Saved → decodelabs_knn_diagnostics_confusion.png
  ✓ Main Plot Saved → decodelabs_knn_diagnostics.png

  🖥️ Displaying Interactive Plot... Close window to continue...

╔══════════════════════════════════════════════════════════════════════════════╗
║                      🎉 MISSION ACCOMPLISHED! 🎉                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║    ⭐ Model: KNeighborsClassifier(n_neighbors=5)                             ║
║    ⭐ Accuracy: 96.67%                                                        ║
║    ⭐ Duration: 3.45 seconds                                                 ║
║    ⭐ Plots Generated: 4 files                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
📋 Table of Contents
Project Overview
Key Features
Technical Stack
Architecture Diagram
Project Structure
Installation & Setup
Configuration (.env)
How to Run
Interactive Features
IPO Pipeline Breakdown
Critical ML Concepts Applied
Results & Metrics
Visualizations
Screenshots
Learning Outcomes
Acknowledgements
🎯 Project Overview
This project implements a production-grade Machine Learning classification pipeline using the classic Iris Dataset to classify iris flowers into three species:
🌸 Iris Setosa
🌼 Iris Versicolor
🌺 Iris Virginica
What Makes This ULTRA:
Table
Feature	Standard	VVIP ULTRA
Terminal UI	Static text	🎬 Animated progress bars + spinners + colored boxes
Visualizations	Static PNG	🎨 4 outputs: Dashboard + Animated GIF + Fancy CM + Main
Themes	Single	🌈 Cyberpunk + Matrix + Ocean (switchable via .env)
Plot Display	Save only	🖥️ Interactive plt.show() with live zoom/pan/save
Error Handling	Basic print	🛡️ Custom exceptions + logging + graceful degradation
Config	Hardcoded	🔐 Secure .env with python-dotenv
Dataset Specifications:
Table
Property	Value
Source	UCI Machine Learning Repository / Ronald Fisher (1936)
Samples	150
Features	4 (sepal length, sepal width, petal length, petal width)
Classes	3 (Setosa, Versicolor, Virginica)
Class Distribution	50 samples per class (perfectly balanced)
✨ Key Features
🎨 Visual & UI Features
Table
Feature	Description	Status
🎬 Animated Progress Bars	Real-time K evaluation with percentage fill	✅
⠋ Spinning Loaders	Cinematic loading animations for each phase	✅
✅ Success Boxes	Green checkmark notifications	✅
⚠️ Warning Boxes	Yellow warning notifications	✅
🌈 3 Color Themes	Cyberpunk (pink/cyan), Matrix (green), Ocean (blue)	✅
🖥️ Interactive Plots	plt.show() opens live matplotlib window	✅
🎞️ Animated GIF Export	Elbow curve draws itself frame-by-frame	✅
🔧 Technical Features
Table
Feature	Description	Status
🔐 Secure Config	Environment variables via .env — zero hardcoded values	✅
🛡️ Error Shielding	Custom exception hierarchy + top-level try-except	✅
📝 Auto-Logging	Dual-channel: Console + Rotating File (logs/app.log)	✅
⚡ Performance	Pre-configured backend, type hints, dataclasses	✅
✅ IPO Pipeline	Clean Input → Process → Output separation	✅
✅ Data Leakage Prevention	Split-before-Scale methodology (Gatekeeper Rule)	✅
✅ Stratified Sampling	80/20 split preserving class proportions	✅
✅ Feature Standardization	StandardScaler (zero mean, unit variance)	✅
✅ Elbow Method	Visual K evaluation (K=1 to K=30)	✅
✅ Blueprint Compliance	Strict K=5 final model	✅
✅ Confusion Matrix	Detailed heatmap with diagonal highlights	✅
✅ Classification Report	Precision, Recall, F1-Score per class	✅
✅ Accuracy Mirage Warning	Educational note on imbalanced data	✅
✅ Reproducibility	Fixed random_state=42	✅
🛠️ Technical Stack
plain
Python 3.10+
├── Standard Library
│   ├── os, sys, time, warnings, logging, logging.handlers
│   ├── dataclasses, datetime, pathlib
│   └── typing (Dict, List, Tuple, Optional, Final)
│
├── Third-Party
│   ├── numpy               → Numerical computations
│   ├── matplotlib          → Data visualization + animation
│   ├── matplotlib.gridspec → Multi-panel layouts
│   ├── matplotlib.animation → Animated GIF generation
│   ├── matplotlib.patches  → Fancy shapes (circles, boxes)
│   ├── seaborn             → Statistical heatmaps
│   ├── scikit-learn        → ML algorithms, preprocessing, metrics
│   └── python-dotenv       → Secure configuration management
│
└── Design Patterns
    ├── IPO Architecture
    ├── Strategy (Theme switching)
    ├── Factory (Visualization Engine)
    └── DTO (Immutable dataclasses)
🏗️ Architecture Diagram
plain
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT PHASE (Animated)                         │
│  ⠋ Loading Dataset...                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │  Load Dataset   │→│  80/20 Split    │→│  StandardScaler │       │
│  │  (load_iris)    │  │  (stratify=y)   │  │  fit_transform  │       │
│  │  150 samples    │  │  random_state=42│  │  on Train only  │       │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘       │
│                              ↓                                         │
│              ┌─────────────────────────────┐                           │
│              │   DatasetMeta + SplitData   │  (Immutable dataclasses)  │
│              └─────────────────────────────┘                           │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                      PROCESS PHASE (Progress Bar)                        │
│  ▸ Elbow Evaluation: [████████████████████░░░░░░░░░░] 75.0%             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Elbow Method (K=1 to K=30)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │  KNN Temp   │→│  Predict    │→│  Error Rate   │            │   │
│  │  │  (loop)     │  │  (on Test)  │  │  (1-Accuracy) │            │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  ⠋ Training Final Model (K=5)...                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │           Final Model: KNeighborsClassifier(n_neighbors=5)    │   │
│  │              Instantiate → Fit → Predict                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                    OUTPUT PHASE (4 Visualizations)                       │
│  ⠋ Generating Ultra Visualizations...                                    │
│  ┌─────────────────────┐  ┌─────────────────────┐                     │
│  │  📊 Dashboard       │  │  🎞️ Animated GIF    │                     │
│  │  (2x2 Grid)         │  │  (Elbow Curve)      │                     │
│  ├─────────────────────┤  ├─────────────────────┤                     │
│  │  🧠 Fancy CM        │  │  📈 Main Plot       │                     │
│  │  (Styled Heatmap)   │  │  (Combined)         │                     │
│  └─────────────────────┘  └─────────────────────┘                     │
│                              ↓                                         │
│  🖥️ Displaying Interactive Plot...                                     │
│  [matplotlib window opens with live, zoomable, panable charts]        │
└─────────────────────────────────────────────────────────────────────────┘
📁 Project Structure
plain
DecodeLabs-Week2-KNN-Iris-ULTRA/
│
├── 📄 README.md                          ← You are here
├── 📄 decodelabs_week2_ultra_fixed.py    ← Main ULTRA application (FIXED)
├── 📄 .env                               ← Environment configuration (DO NOT UPLOAD TO GITHUB!)
├── 📄 .gitignore                         ← Excludes .env, logs/, *.log, *.png, *.gif
├── 📄 requirements.txt                   ← Python dependencies
│
├── 📂 logs/
│   └── 📄 app.log                        ← Auto-generated audit trail
│
├── 📄 decodelabs_knn_diagnostics.png              ← Main combined plot
├── 📄 decodelabs_knn_diagnostics_dashboard.png    ← 2x2 dashboard
├── 📄 decodelabs_knn_diagnostics_animated.gif     ← Animated elbow curve
├── 📄 decodelabs_knn_diagnostics_confusion.png    ← Fancy confusion matrix
│
└── 📂 assets/
    ├── 📸 screenshot_terminal.png        ← Animated terminal output
    ├── 📸 screenshot_dashboard.png       ← 2x2 dashboard plot
    ├── 📸 screenshot_animation.gif       ← Animated elbow curve
    ├── 📸 screenshot_confusion.png       │ Fancy confusion matrix
    └── 📸 screenshot_interactive.png     ← plt.show() live window
⚙️ Installation & Setup
Step 1: Clone or Download
bash
git clone https://github.com/yourusername/DecodeLabs-Week2-KNN-Iris-ULTRA.git
cd DecodeLabs-Week2-KNN-Iris-ULTRA
Step 2: Create Virtual Environment (Recommended)
bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install numpy matplotlib seaborn scikit-learn python-dotenv pillow
Note: pillow is required for animated GIF export. If not installed, animation will be skipped gracefully with a warning message.
Or use the requirements file:
bash
pip install -r requirements.txt
Step 4: Create .env File
Create a file named .env in the same directory:
env
# ── ML Hyperparameters ──────────────────
RANDOM_STATE=42
TEST_SIZE=0.20
STANDARD_K=5
K_MIN=1
K_MAX=31

# ── Visualization ───────────────────────
FIGURE_DPI=150
FIGURE_WIDTH=16
FIGURE_HEIGHT=10
OUTPUT_PLOT=decodelabs_knn_diagnostics.png
ANIMATION_FPS=8
THEME=cyberpunk        # Options: cyberpunk | matrix | ocean
SHOW_PLOT=True         # Set to False to skip plt.show()

# ── Logging ─────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_DIR=logs
⚠️ IMPORTANT: Add .env to your .gitignore!
gitignore
# .gitignore
.env
logs/
*.log
__pycache__/
venv/
*.png
*.gif
🚀 How to Run
Basic Run (Cyberpunk Theme)
bash
python decodelabs_week2_ultra_fixed.py
Change Theme
Edit .env:
env
THEME=matrix      # Green hacker theme
# or
THEME=ocean       # Blue ocean theme
# or
THEME=cyberpunk   # Pink/cyan neon theme (default)
Disable Interactive Plot (Headless/Server)
env
SHOW_PLOT=False
🎮 Interactive Features
🎬 Terminal Animations
Table
Animation	When It Appears	Description
Spinner ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏	Dataset loading, model training, visualization	Unicode braille spinner
Progress Bar ████░░░░	Elbow evaluation (K=1 to 30)	Real-time percentage fill with color
Success Box ✓	After each completed step	Green checkmark with message
Warning Box ⚠	Educational warnings	Yellow warning with message
🌈 Color Themes
Cyberpunk (Default)
plain
Background: #0a0a0f (Dark)
Primary:    #ff00ff (Hot Pink)
Secondary:  #00ffff (Cyan)
Accent:     #ffff00 (Yellow)
Matrix
plain
Background: #000000 (Black)
Primary:    #00ff00 (Matrix Green)
Secondary:  #55ff55 (Light Green)
Accent:     #aaffaa (Lime)
Ocean
plain
Background: #001020 (Deep Blue)
Primary:    #00aaff (Ocean Blue)
Secondary:  #44ddff (Light Blue)
Accent:     #88eeff (Pale Blue)
🖥️ Interactive Plot Features
When SHOW_PLOT=True:
✅ Live matplotlib window opens automatically
✅ Zoom — Scroll mouse wheel to zoom in/out
✅ Pan — Click and drag to explore different regions
✅ Save — Click floppy disk icon to save from GUI
✅ 4 subplots — Elbow curve, confusion matrix, metrics bar, class pie — all interactive!
🎞️ Animated GIF
The animated elbow curve:
Draws itself point-by-point (frame-by-frame)
Shows current K value and error rate in real-time
Highlights blueprint K=5 with dashed orange line
Saves as decodelabs_knn_diagnostics_animated.gif
🔬 IPO Pipeline Breakdown
📥 INPUT PHASE (with Spinner Animation)
Python
# 1. Load Dataset with animation
# Shows: ⠋ Loading Iris Dataset... → ✓ complete!

# 2. Stratified Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# 3. StandardScaler (Gatekeeper Rule)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit ONLY on train
X_test_scaled  = scaler.transform(X_test)          # Transform test
⚙️ PROCESS PHASE (with Progress Bar)
Python
# Elbow Method with animated progress bar
for k in range(1, 31):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    error_rates.append(1 - accuracy_score(y_test, knn_temp.predict(X_test_scaled)))
    # Updates: ▸ Elbow Evaluation: [████░░░░] 25.0%

# Final Model (K=5) with spinner
# Shows: ⠋ Training Final Model (K=5)... → ✓ complete!
model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
model.fit(X_train_scaled, y_train)
📤 OUTPUT PHASE (4 Visualizations)
Python
# 1. Dashboard (2x2 grid) — PNG
# 2. Animated Elbow Curve — GIF
# 3. Fancy Confusion Matrix — PNG
# 4. Main Combined Plot — PNG
# 5. Interactive plt.show() window — LIVE
🧠 Critical ML Concepts Applied
1️⃣ Data Leakage Prevention (The Gatekeeper Rule)
Problem: Scaling before splitting leaks test data statistics (mean/variance) into training.
Solution: fit_transform() on Train → transform() on Test only.
Impact: Prevents inflated accuracy scores in production.
2️⃣ Stratified Sampling
Ensures each class (Setosa, Versicolor, Virginica) maintains its original proportion (33.3% each) in both train and test sets. Without stratification, random split might give 40% Setosa in train and 20% in test — biasing the model.
3️⃣ Feature Scaling (Standardization)
KNN is a distance-based algorithm. Unscaled features (e.g., petal length in cm vs. sepal width in mm) would dominate distance calculations. StandardScaler converts all features to:
Mean = 0
Variance = 1
This ensures equal contribution from all features regardless of original scale.
4️⃣ Elbow Method vs. Blueprint Compliance
While the Elbow Method evaluates K=1 to K=30 for visualization and educational purposes, the final model strictly uses K=5 as per project instructions. This demonstrates:
Understanding of hyperparameter tuning methodology
Ability to follow strict technical specifications
Separation of exploration vs. production model
5️⃣ Reproducibility
Fixed random_state=42 ensures:
Identical train-test splits every run
Consistent results for peer review and debugging
Reliable CI/CD pipeline integration
6️⃣ Multi-Metric Evaluation (Beyond Accuracy)
Accuracy Mirage Warning: On imbalanced datasets, a model predicting only the majority class can score >95% accuracy while achieving 0% recall on minority classes.
Solution: Always validate with Confusion Matrix + F1-Score.
📊 Results & Metrics
Overall Accuracy
plain
⭐ Overall Accuracy : 96.67% - 100.00%
(May vary slightly based on random split, but typically 96-100% on Iris)
Confusion Matrix
Table
Predicted Setosa	Predicted Versicolor	Predicted Virginica
Actual Setosa	10	0	0
Actual Versicolor	0	10	0
Actual Virginica	0	1	9
Interpretation:
Setosa: Perfect classification (10/10 correct)
Versicolor: Perfect classification (10/10 correct)
Virginica: 1 misclassification (predicted as Versicolor)
Total Errors: 1 out of 30 test samples
Classification Report
plain
                 precision    recall  f1-score   support

    setosa         1.00        1.00      1.00        10
versicolor         0.91        1.00      0.95        10
 virginica         1.00        0.90      0.95        10

    accuracy                             0.97        30
   macro avg         0.97        0.97      0.97        30
weighted avg         0.97        0.97      0.97        30
Key Terms:
Precision: Of all predicted as X, how many were actually X?
Recall: Of all actual X, how many were correctly predicted?
F1-Score: Harmonic mean of Precision & Recall (balances both)
📈 Visualizations
4 Output Files Generated:
Table
File	Description	Format
decodelabs_knn_diagnostics.png	Main combined plot	PNG
decodelabs_knn_diagnostics_dashboard.png	2x2 dashboard (Elbow + CM + Metrics + Pie)	PNG
decodelabs_knn_diagnostics_animated.gif	Animated elbow curve drawing	GIF
decodelabs_knn_diagnostics_confusion.png	Fancy confusion matrix with diagonal highlights	PNG
Dashboard Layout (2x2):
plain
┌─────────────────┬─────────────────┐
│  📉 Elbow Curve │  🧠 Confusion   │
│  (K vs Error)   │     Matrix      │
├─────────────────┼─────────────────┤
│  📊 Metrics Bar │  🌸 Class Pie   │
│  (Accuracy etc) │   (Distribution)│
└─────────────────┴─────────────────┘
📸 Screenshots
Add your execution screenshots in the /assets/ folder and reference them here:
Table
Screenshot	Description
Animated terminal with progress bars and spinners
2x2 dashboard visualization
Animated elbow curve GIF
Fancy confusion matrix with diagonal highlights
Live plt.show() window with zoom/pan
🎓 Learning Outcomes
Through this project, I have demonstrated:
✅ Supervised Classification — Complete ML workflow from raw data to prediction
✅ Data Leakage Prevention — Critical preprocessing discipline (Split-Before-Scale)
✅ Stratified Sampling — Maintaining class distribution integrity
✅ Feature Engineering — StandardScaler normalization for distance-based algorithms
✅ Scikit-Learn Proficiency — fit/predict pattern, preprocessing pipelines, metrics API
✅ Hyperparameter Awareness — Elbow Method exploration vs. specification compliance
✅ Model Evaluation — Multi-metric validation (Accuracy, Precision, Recall, F1, CM)
✅ Visualization Mastery — Matplotlib animations, multi-panel dashboards, themes
✅ Reproducibility — Fixed random seeds and deterministic workflows
✅ IPO Architecture — Clean separation of Input, Process, and Output phases
✅ Secure Configuration — python-dotenv externalization with runtime validation
✅ Error Handling — Custom exception hierarchy with graceful degradation
✅ Logging Discipline — Dual-channel structured logging for audit and debug
✅ Type Safety — Comprehensive type hints and immutable dataclasses
✅ UI/UX Design — Terminal animations, progress indicators, color themes
🙏 Acknowledgements
DecodeLabs — For providing this structured internship and learning opportunity
Scikit-Learn Team — For the robust, well-documented ML framework
UCI Machine Learning Repository — For hosting the classic Iris dataset
Ronald Fisher — Original collector and publisher of the Iris dataset (1936)
Matplotlib Community — Animation and visualization capabilities
python-dotenv — Secure configuration management
📬 Contact
For any queries regarding this project:
Intern Name: [M AHMED ALI ]
Email: [muhammadahmedali607@gmail.com ]
LinkedIn: [  www.linkedin.com/in/muhammad-ahmed-ali-123125406]
GitHub: [https://github.com/mrahmed121]
<div align="center">
⭐ DecodeLabs AI Engineering Internship — Week 2 ⭐
Submitted as part of the official internship program.
</div>
