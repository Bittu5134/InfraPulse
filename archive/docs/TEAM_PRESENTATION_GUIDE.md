# 🏛️ InfraPulse — Complete Team Presentation & Architecture Guide

> **Welcome Team!**  
> This guide is crafted specifically for you. Even if you didn't write the code, this document will give you a **100% crystal-clear understanding** of how InfraPulse works under the hood, how the website communicates with the server, why every engineering decision was made, and how to confidently present and defend this project in front of judges!

---

## 📑 Table of Contents
1. [The 30-Second Elevator Pitch](#1-the-30-second-elevator-pitch)
2. [The Core Problem & Our Solution](#2-the-core-problem--our-solution)
3. [Jargon Buster: Every Technical Term Explained Simply](#3-jargon-buster-every-technical-term-explained-simply)
4. [High-Level System Architecture](#4-high-level-system-architecture)
5. [The Life of a Defect Report: How Frontend & Backend Talk](#5-the-life-of-a-defect-report-how-frontend--backend-talk)
6. [Real-Time Magic: How Live Updates Work (SSE)](#6-real-time-magic-how-live-updates-work-sse)
7. [The AI Engine & Decision-Making (No Text Allowed!)](#7-the-ai-engine--decision-making-no-text-allowed)
8. [The Priority Ranking Formula: The Math Behind the Queue](#8-the-priority-ranking-formula-the-math-behind-the-queue)
9. [User Roles & The 5 Portals](#9-user-roles--the-5-portals)
10. [Step-by-Step Demo Script for Judges (3-Minute Presentation)](#10-step-by-step-demo-script-for-judges-3-minute-presentation)
11. [Judge Q&A Defense Strategy (Tough Questions & Winning Answers)](#11-judge-qa-defense-strategy-tough-questions--winning-answers)

---

## 1. The 30-Second Elevator Pitch

> *"InfraPulse is an intelligent, photo-based infrastructure defect triage and maintenance queue system. When a citizen takes a photo of damage—like concrete spalling, stagnant water, cracked tiles, or peeling paint—our computer vision ensemble automatically verifies image quality, classifies the defect, measures visible severity and damage area, and places the ticket into a live, mathematically ranked queue for dedicated repair teams in real-time. There is zero human guesswork, zero text-based gaming of the system, and 100% objective, transparent prioritization."*

---

## 2. The Core Problem & Our Solution

### ❌ The Old Way (How Facility Maintenance Usually Fails)
* **First-Come, First-Served Chaos**: A cosmetic paint scratch reported on Monday gets fixed before a dangerous concrete ceiling collapse reported on Tuesday.
* **Subjective Panic**: Users type *"EMERGENCY!! LIFE THREATENING!"* for minor issues to get attention, causing staff fatigue.
* **Manual Routing Bottlenecks**: A single admin has to manually open hundreds of tickets and figure out which department (civil, plumbing, sanitation) to send them to.
* **Zero Transparency**: Citizens submit complaints and have no idea where their issue stands in line.

### ✅ The InfraPulse Way
* **100% Automated Triage**: The AI analyzes raw pixels—not user exaggeration.
* **Objective Math-Based Queues**: Dangerous structural hazards automatically jump ahead of cosmetic flaws based on scientific severity and spatial area calculations.
* **Instant Department Routing**: Structural, Functional, and Performance tickets go directly to dedicated staff portals without manual sorting.
* **Live Dynamic Tracking**: Citizens can watch their exact queue position update in real time without refreshing their browser.

---

## 3. Jargon Buster: Every Technical Term Explained Simply

Whenever judges ask technical questions, use these intuitive analogies:

| Technical Term | What It Is (Simple Explanation) | Everyday Analogy |
| :--- | :--- | :--- |
| **Frontend** | What the user sees on their screen (buttons, forms, cards, live meters, styling). Built with HTML, CSS (Tailwind), and JavaScript. | **The Dining Area of a Restaurant** — the tables, menus, decor, and the interface the customer interacts with. |
| **Backend** | The hidden computer server running our Python logic, database queries, and AI models. Built with **FastAPI**. | **The Kitchen of a Restaurant** — where the chefs cook the food, manage ingredients, and execute recipes. |
| **HTTP Request (`GET` / `POST`)** | How the frontend asks the backend for something (`GET` = fetch data, `POST` = send new data like an uploaded photo). | **The Waiter taking your order** from the table to the kitchen. |
| **API (Application Programming Interface)** | Specific numbered "counter windows" or URL endpoints where the frontend talks to the backend (e.g., `/api/submit`). | **The Drive-thru window** with a strict menu of what you can order and receive. |
| **JSON (JavaScript Object Notation)** | A clean, universal text format for sending structured data between frontend and backend (e.g., `{"defect": "spalling", "severity": 8.5}`). | **A Standardized Order Slip** that both the waiter and chef can read without miscommunication. |
| **Session Cookie / Token** | A small piece of secure data stored in the user's browser after logging in, proving who they are on future page visits. | **A VIP Wristband or Hand Stamp** at a concert so you don't have to show your ID ticket every time you re-enter. |
| **FastAPI** | The ultra-fast Python web framework powering our backend server. It handles web traffic and connects to the AI models. | **An efficient head chef & order coordinator** directing kitchen operations at lightning speed. |
| **SQLite Database** | The lightweight, file-based database (`infrapulse.db`) where all user accounts, complaint tickets, timestamps, and scores are safely stored. | **The Master Filing Cabinet** holding organized records of all current and historical maintenance tickets. |
| **Server-Sent Events (SSE)** | A technology that lets the server send instant updates down to the browser over an open connection without the user having to press reload (F5). | **An Airport Live Flight Departure Board** that updates gate changes and delays automatically every second. |
| **PyTorch** | The industry-standard deep learning / AI library in Python used to run our computer vision neural networks. | **The Brain Simulation Engine** used to inspect and classify images. |
| **AI Ensemble** | Combining multiple distinct AI models together to vote on the result instead of relying on just one single model. | **A Panel of 5 Specialist Doctors** consulting together before giving a final medical diagnosis. |
| **Quality Gate** | An automated pre-check that tests whether an uploaded photo is too blurry or too dark before letting the AI analyze it. | **A Security Guard at the entrance** checking that your ID photo is sharp and readable before letting you in. |
| **Sobel Edge Detection** | A mathematical image-processing technique that calculates sharp brightness changes to highlight cracks, peeling edges, and holes. | **A Fluorescent Highlighter** tracing out the exact outline of physical cracks on a wall. |
| **Dilation Neighborhood** | Expanding detected crack lines by a few pixels so thin cracks are clearly visible to human inspectors on screen. | **Switching from a fine pen to a thick marker** so the highlighted crack pops out on any screen. |
| **Tier Base Points (3000 / 2000 / 1000)** | Fixed large point offsets added to tickets according to danger category so critical structural threats can never be overtaken by minor cosmetic issues. | **Emergency Room Triage Tiers**: Cardiac Arrest (3000 pts) > Broken Arm (2000 pts) > Paper Cut (1000 pts). |

---

## 4. High-Level System Architecture

Here is how all the pieces of InfraPulse fit together in one complete loop:

```mermaid
flowchart TD
    subgraph ClientLayer ["1. FRONTEND LAYER (Browser)"]
        UI_User["Citizen Reporter Portal<br>(/user/submit)"]
        UI_Track["Citizen Live Tracker<br>(/user/track)"]
        UI_Staff["Staff Queue Portals<br>(/staff/queue)"]
    end

    subgraph ServerLayer ["2. FASTAPI BACKEND SERVER"]
        Router["FastAPI Request Router & Auth"]
        QGate["Quality Gate Engine<br>(Blur & Brightness Check)"]
        SSE["SSE Live Stream Broadcaster<br>(/live/queue_stream)"]
    end

    subgraph AILayer ["3. VISION AI & MATH ENGINE"]
        subgraph Ensemble ["5-Model Vision Ensemble"]
            M1["ConvNeXt-Tiny (Modern CNN)"]
            M2["Swin-Transformer (Attention)"]
            M3["MTL Dual-Branch (Rebar/Spalling)"]
            M4["Baseline EfficientNet"]
            M5["Quantized INT8 Engine"]
        end
        Consensus["Per-Category Weighted Soft Voting<br>(consensus_weights.json)"]
        Spatial["Sobel Edge Dilation & Contrast Math<br>(Calculates Severity & Extent)"]
        PriorityCalc["Priority Ranking Formula<br>TierBase + Sev*5 + Ext*3 + TimeBonus"]
    end

    subgraph DataLayer ["4. PERSISTENCE LAYER"]
        DB[(SQLite Database<br>infrapulse.db)]
    end

    %% Flow Connections
    UI_User -->|1. HTTP POST with Photo| Router
    Router -->|2. Check Quality| QGate
    QGate -->|3. Valid Image| Ensemble
    Ensemble -->|4. Raw Predictions| Consensus
    Consensus -->|5. Defect Class & Category| Spatial
    Spatial -->|6. Severity (0-10) & Extent (0-10)| PriorityCalc
    PriorityCalc -->|7. Final Priority Score| DB
    DB -->|8. Push Queue Update| SSE
    SSE -->|9. Real-Time Stream Events| UI_Track
    SSE -->|9. Real-Time Stream Events| UI_Staff
```

---

## 5. The Life of a Defect Report: How Frontend & Backend Talk

Let's follow the complete story of what happens when a citizen reports a cracked ceiling or broken tile:

```mermaid
sequenceDiagram
    autonumber
    actor Citizen as Citizen (Browser)
    participant Front as Frontend UI (HTML/JS)
    participant Back as FastAPI Backend
    participant QG as Quality Gate
    participant AI as 5-Model AI Ensemble
    participant Math as Spatial & Priority Engine
    participant DB as SQLite Database
    actor Staff as Maintenance Staff

    Citizen->>Front: 1. Selects photo & location
    Front->>Front: 2. Instant HTML5 Canvas Preview
    Citizen->>Front: 3. Clicks "Submit Complaint"
    Front->>Back: 4. HTTP POST /user/submit (Multipart Form + Image)
    
    Back->>QG: 5. Verify image quality (Laplacian Blur & Brightness)
    alt Image is blurry/dark
        QG-->>Back: Quality Failure
        Back-->>Front: Error: Please re-take clear photo
    else Image is valid
        QG-->>AI: 6. Pass image pixels to AI Ensemble
        AI->>AI: 7. 5 Models vote with weighted consensus matrix
        AI-->>Math: 8. Defect class (e.g., 'spalling') & Confidence
        Math->>Math: 9. Compute Sobel Edge Density (Severity) & Anomaly Area (Extent)
        Math->>Math: 10. Calculate Score: Base(3000) + Sev*5 + Ext*3 + TimeBonus
        Math-->>DB: 11. INSERT new ticket into SQLite database
        DB-->>Back: 12. Ticket ID & Queue Position
        Back-->>Front: 13. Redirect to /user/ticket/{id} with Live Tracking
        Back-)Staff: 14. Real-time SSE Broadcast updates Staff Queue
    end
```

### Step-by-Step Breakdown:

#### 1. In the Browser (Frontend):
* The citizen visits `/user/submit`.
* When they pick a photo, JavaScript immediately displays an **instant client-side preview** using an HTML5 Canvas so the user knows they selected the right file.
* When they hit submit, the browser packages the photo file, latitude/longitude, and location text into an **HTTP `POST` multipart form request** and sends it over the internet to the backend.

#### 2. In the FastAPI Server (Backend):
* FastAPI receives the request and immediately sends the image to the **Quality Gate** (`app/quality_gate.py`).
* The Quality Gate runs two quick mathematical checks:
  1. **Blur Check (Laplacian Variance)**: Calculates sharpness. If variance is $< 100$, the image is rejected as too blurry.
  2. **Exposure Check**: Calculates average pixel brightness. If mean $< 25$ (pitch black) or $> 235$ (completely washed out), it is rejected.

#### 3. In the AI Ensemble (`app/model/src/inference.py`):
* If the image passes, it is resized to $224 \times 224$ pixels and fed into our **5-model vision ensemble**:
  1. **ConvNeXt-Tiny** (State-of-the-art pure convolutional network)
  2. **Swin-Transformer (Swin-T)** (Uses attention mechanisms to look at global patterns)
  3. **Multi-Task Learning (MTL) Dual-Branch** (Specialist in structural spalling and exposed rebar)
  4. **EfficientNet-B0 Baseline**
  5. **INT8 Quantized Fast Engine**
* The models' outputs are combined using our **Calibrated Per-Category Weight Matrix** (`consensus_weights.json`) to find the winning defect category:
  * `spalling` $\rightarrow$ **Structural** Category
  * `stagnant_water` $\rightarrow$ **Functional** Category
  * `cracked_tiles` $\rightarrow$ **Performance** Category
  * `paint_peeling` $\rightarrow$ **Performance** Category

#### 4. Spatial Edge & Extent Calculation (`app/priority_queue.py`):
* The system computes the **Sobel Spatial Gradient**:
  $$\text{GradientMagnitude} = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}$$
* This detects how sharp and dense the cracks/damage borders are $\rightarrow$ Yields **Visible Severity (0.0 to 10.0)**.
* It calculates the percentage of the image occupied by abnormal contrast and edges $\rightarrow$ Yields **Visible Extent (0.0 to 10.0)**.

#### 5. Priority Score Calculation & Database Storage:
* The backend computes the final score using our strict priority formula.
* The ticket is inserted into `infrapulse.db` with status `Submitted`.
* The server sends a response back to the citizen's browser, redirecting them to their personalized tracking dashboard.

---

## 6. Real-Time Magic: How Live Updates Work (SSE)

A key highlight that judges love is **real-time synchronization**. How do staff and citizens see live updates without ever clicking refresh?

```mermaid
flowchart LR
    A[Staff clicks 'Resolve' or 'Assign'] -->|HTTP POST| B[FastAPI Backend /staff/update_status]
    B -->|Update DB record| C[(SQLite DB)]
    B -->|Trigger Broadcast Event| D[SSE Stream Manager /live/queue_stream]
    D -->|Push JSON Data| E[Citizen Track Screen - Badge flips to 'Assigned']
    D -->|Push JSON Data| F[Staff Live Queue - Ticket updates in real-time]
```

### Why Server-Sent Events (SSE) instead of WebSockets or Polling?
* **Why not constant Polling (F5 auto-refresh every 2 seconds)?**  
  Polling wastes bandwidth, floods the server with useless requests, and drains mobile battery.
* **Why not WebSockets?**  
  WebSockets are two-way (bidirectional) and heavy to maintain behind corporate firewalls.
* **Why SSE (Server-Sent Events) is the Perfect Choice**:  
  Our updates are **one-way** (server pushes updates to browsers when ticket status changes). SSE runs over standard HTTP, uses built-in browser reconnection, is lightweight, and works seamlessly on cloud platforms like Railway!

---

## 7. The AI Engine & Decision-Making (No Text Allowed!)

### 🚨 Why We STRICTLY Forbid Text Inputs in AI Priority
Judges will often ask: *"Why didn't you use ChatGPT / NLP to read the user's description?"*

**Your Team's Winning Answer:**
1. **Strict Problem Statement Compliance**: The competition problem statement explicitly commands:
   > *"Classification limited strictly to what is visibly evident in the photograph, no claims about non-visible or predicted defects."*
2. **Anti-Gaming / Security Defense (Prompt Injection)**: If the system read text descriptions, any student or tenant could type *"Catastrophic foundation collapse emergency!"* for a tiny paint scratch to jump the queue. Relying **100% on raw visual pixels** guarantees an unhackable, tamper-proof, objective system.
3. **No Language Barrier**: A citizen who cannot speak English or write elaborate text receives the exact same high-priority service based purely on the photo of the hazard.

### How the Weighted Consensus Works (Doctor Panel Analogy)
Imagine a panel of 5 doctors diagnosing a patient:
* When diagnosing **Cracked Tiles**, Dr. ConvNeXt has a 60% vote and Dr. Swin-T has a 30% vote because they have proven 95%+ precision on tile textures.
* When diagnosing **Spalling (Concrete damage)**, Dr. MTL Specialist has a 35% vote because it was specially trained on exposed metal reinforcement bars.
* Models that showed false hallucinations during testing (e.g. baseline mistaking shiny paint for water) have their votes set to **0.0%** for that category.

$$\text{Consensus Score } S_c = \frac{\sum W(c, m) \cdot P_m(c)}{\sum W(c, m)}$$

---

## 8. The Priority Ranking Formula: The Math Behind the Queue

Every ticket in InfraPulse receives an exact mathematical score:

$$\mathbf{PriorityScore} = \mathbf{CategoryTierBase} + (\mathbf{Severity} \times 5.0) + (\mathbf{Extent} \times 3.0) + \mathbf{CappedTimeBonus}$$

### Breakdown of the 4 Formula Components:

```mermaid
graph LR
    Score["Total Priority Score"]
    
    Score --> A["1. Category Tier Base<br>Structural: 3000 pts<br>Functional: 2000 pts<br>Performance: 1000 pts"]
    Score --> B["2. Severity (0-10) x 5.0<br>Measures crack depth & edge sharpness<br>(Max 50.0 pts)"]
    Score --> C["3. Extent (0-10) x 3.0<br>Measures % area damaged<br>(Max 30.0 pts)"]
    Score --> D["4. Capped Time Bonus<br>0.05 pts per hour<br>(STRICTLY CAPPED at max 5.0 pts)"]
```

#### 1. Category Tier Base (Domain Isolation Safeguard)
* **Structural (`Spalling`) $\rightarrow$ 3000 Base Points**: Structural defects pose life-safety risks (falling concrete, structural failure) and must **always** be above non-structural defects.
* **Functional (`Stagnant Water`) $\rightarrow$ 2000 Base Points**: Flooding, blocked drainage, hygiene hazards.
* **Performance (`Cracked Tiles`, `Paint Peeling`) $\rightarrow$ 1000 Base Points**: Serviceability and cosmetic issues.  
  *(Note: `Cracked Tiles` gets a `+1.0` bonus over `Paint Peeling` because broken tiles are a tripping/cutting hazard).*

#### 2. Severity Multiplier ($\times 5.0$, up to 50 pts)
Extracted via Sobel gradient magnitude. Deep, wide cracks generate high gradients and earn up to 50 points.

#### 3. Extent Multiplier ($\times 3.0$, up to 30 pts)
Extracted via anomaly area coverage. Damage spreading across the entire wall earns up to 30 points.

#### 4. The Capped Time Bonus ($\max 5.0$ pts) — The "Escalation Trap" Prevention
$$\text{TimeBonus} = \min(5.0, \, \text{Age in Hours} \times 0.05)$$

> **Why is it strictly capped at 5.0 points?**  
> If time bonus was uncapped, a trivial paint peeling ticket sitting in the database for 3 months would accumulate hundreds of points and jump above a brand-new, life-threatening ceiling crack!  
> By capping the time bonus at **5.0 points**, it acts **strictly as a tie-breaker** between tickets of similar severity, guaranteeing that critical safety threats always cut to the front of the line!

---

## 9. User Roles & The 5 Portals

InfraPulse provides isolated, secure portals with distinct responsibilities:

| Portal | URL | Demo Login | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Citizen Portal** | `/user/submit` & `/user/dashboard` | `user@infrapulse.org` / `user123` | Report new complaints, track live queue standing, view highlighted defect reticle. |
| **Structural Staff Queue** | `/staff/queue?category=Structural` | `structural@infrapulse.org` / `staff123` | Triage & resolve **Structural** hazards (Spalling, concrete defects) with 3000+ priority base. |
| **Functional Staff Queue** | `/staff/queue?category=Functional` | `functional@infrapulse.org` / `staff123` | Triage & resolve **Functional** hazards (Stagnant water, leaks) with 2000+ priority base. |
| **Performance Staff Queue** | `/staff/queue?category=Performance` | `performance@infrapulse.org` / `staff123` | Triage & resolve **Performance** defects (Cracked tiles, paint peeling) with 1000+ priority base. |
| **Admin Control Center** | `/admin/dashboard` | `admin@infrapulse.org` / `admin123` | Global overview across all categories, workload statistics, staff management. |

---

## 10. Step-by-Step Demo Script for Judges (3-Minute Presentation)

Here is a ready-to-use script for your live presentation:

### ⏱️ Minute 1: The Hook & The Problem
> *"Good morning/afternoon, judges! In campus hostels, public hospitals, and municipal buildings, maintenance requests are usually handled on a first-come, first-served basis or whoever complains the loudest. A critical ceiling spalling hazard can sit unaddressed while maintenance spends three days repainting a hallway.*  
>  
> *We built **InfraPulse** to solve this. InfraPulse is a photo-based, automated defect triage system that uses computer vision and dynamic spatial math to classify defects and rank repair queues objectively with zero human bias."*

### ⏱️ Minute 2: The Live Citizen Flow & AI Inference
> *(Action: Open `/user/submit` and upload a spalling/cracked tile photo)*  
>  
> *"Watch what happens when a citizen reports damage. Notice our client-side instant preview. When we click Submit, the photo enters our FastAPI backend:*  
> *1. First, our **Quality Gate** verifies the photo isn't blurry or dark.*  
> *2. Next, our **5-Model Vision AI Ensemble** performs calibrated weighted consensus to classify the defect as `Spalling` with high confidence.*  
> *3. Our spatial gradient engine measures the exact crack edge density and damage area.*  
> *4. Finally, our priority algorithm assigns a calculated score—here you see it scored **3,068.4 points**, automatically placing it in Tier 1 (Structural)."*

### ⏱️ Minute 3: Real-Time Staff Queue & Instant Sync
> *(Action: Open `/staff/queue` in a second side-by-side browser window)*  
>  
> *"Notice how the ticket was routed directly to the Structural Staff Queue without any manual human sorting. As the maintenance technician accepts the ticket and marks it `In Progress`, look at the citizen's screen on the left—thanks to our **Server-Sent Events (SSE)** architecture, the citizen's badge flips to `In Progress` in real time without refreshing the page!*  
>  
> *InfraPulse delivers total transparency, mathematically optimal repair scheduling, and safer public infrastructure. Thank you, and we'd love to take your questions!"*

---

## 11. Judge Q&A Defense Strategy (Tough Questions & Winning Answers)

Be prepared for these classic judge questions:

### ❓ Q1: "Why did you use 5 models instead of just one?"
> **Winning Answer:**  
> *"Single models often suffer from category-specific blind spots. For instance, in our testing, standard EfficientNet frequently confused shiny reflective paint with stagnant water. By combining ConvNeXt-Tiny, Swin Transformer, and a specialized Multi-Task Learning model with our **Calibrated Per-Category Consensus Matrix**, we achieved **91.29% accuracy** and a **100% recall on stagnant water**, while keeping inference latency at just ~200ms."*

### ❓ Q2: "What happens if a user uploads a completely blurry or dark photo?"
> **Winning Answer:**  
> *"InfraPulse has a dedicated **Quality Gate** built on Laplacian variance and luminance thresholding. If an image has a blur variance below 100 or brightness outside the 25-235 range, the system rejects it immediately and asks the user to retake a clear photo before running expensive AI inference."*

### ❓ Q3: "How do you ensure a cosmetic complaint doesn't jump ahead of a dangerous ceiling crack?"
> **Winning Answer:**  
> *"We enforce strict **Category Tier Base Points**: Structural issues automatically start with 3000 base points, Functional issues start with 2000, and Performance issues start with 1000. Because the maximum points earned from severity and extent combined is 80 points, a cosmetic issue (max ~1080 pts) can mathematically never overtake even the mildest structural issue (min 3000 pts)."*

### ❓ Q4: "What if a cosmetic issue sits in the database for 6 months? Will it ever overtake a new structural issue?"
> **Winning Answer:**  
> *"No! We specifically engineered a **Strictly Capped Time Bonus** (maximum 5.0 points). The time bonus is designed solely as a tie-breaker between complaints in the same tier and severity level. It will never allow an old cosmetic complaint to climb into the structural queue."*

### ❓ Q5: "How does the live updates feature scale compared to WebSockets?"
> **Winning Answer:**  
> *"We utilized **Server-Sent Events (SSE)**. Since status updates are one-directional (from backend to frontend), SSE is vastly lighter on server memory than WebSockets, handles automatic reconnections natively in HTTP, and easily scales on cloud containers without firewall dropouts."*

### ❓ Q6: "Why didn't you let users type in descriptions to boost their priority?"
> **Winning Answer:**  
> *"Two reasons: First, the competition problem statement mandates classification strictly from visible evidence. Second, allowing text descriptions introduces severe security and prompt injection vulnerabilities where users can exaggerate minor issues to manipulate queue rankings. Our pixel-based inference is 100% objective and tamper-proof."*

---

## 🚀 Quick Verification Commands (If Judges Ask to See Tests)

If a judge asks *"Did you write automated tests for this?"*, you can show them our clean test suite:

```bash
python -m pytest tests/ -v
```
**Expected Output:**
* `test_priority_score_computation` $\rightarrow$ **PASSED** (Validates formula & tier boundaries)
* `test_user_registration_login_and_ticket_submission` $\rightarrow$ **PASSED** (Validates full user flow)
* `test_staff_login_and_self_assignment` $\rightarrow$ **PASSED** (Validates queue triage)
* `test_admin_portal_management` $\rightarrow$ **PASSED** (Validates admin oversight)
* `test_benchmark_page` & `test_custom_playground` $\rightarrow$ **PASSED** (Validates live evaluation tools)

---

*All the best for your presentation! You have an airtight, mathematically sound, and beautifully engineered system. Go win those judges! 🏆*
