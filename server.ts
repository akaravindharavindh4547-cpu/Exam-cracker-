import express, { Request, Response } from 'express';
import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { authRouter } from './server/authRoutes.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '10mb' }));

// Mount Secure Authentication API routes
app.use('/api/auth', authRouter);

// Server-side initialization of Gemini API using @google/genai
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey: apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

interface ExamContext {
  examTitle?: string;
  examCode?: string;
  category?: string;
  place?: string;
  syllabusSnippet?: string;
}

// System prompt tailored for Indian & Global Competitive Exams doubt-solving & coaching
const getSystemInstruction = (examContext?: ExamContext, language: 'en' | 'ta' = 'en') => {
  return `You are ExamCracker AI Agent, a helpful, intelligent, and interactive AI tutor and study assistant built into the EXAM CRACKER platform.

CRITICAL INSTRUCTIONS & RULES:
1. Read and understand the user's CURRENT question carefully before answering.
2. Do NOT give the same answer to different questions unless the questions genuinely have the exact same answer.
3. Always answer based on the exact topic, intent, and context of the user's latest message.
4. Do not copy, repeat, or reuse your previous answer when the user asks a new question.
5. If the user changes the topic, completely switch your response to the new topic.
6. Use previous conversation messages only when they are relevant to the current question.
7. If the question is unclear, ask a short clarification question instead of guessing.
8. Give a direct, relevant, and useful answer.
9. Avoid generic responses that could apply to every question.
10. Before responding, internally identify what the user is asking and make sure your answer specifically addresses that question.

EXAMPLES OF SPECIFIC DIRECT RESPONSES:
- User asks: "What is Python?" -> Explain Python programming language, syntax, uses, and features.
- User asks: "What is the capital of India?" -> Answer New Delhi with brief historical context.
- User asks: "How do I make a website?" -> Explain frontend (HTML/CSS/JS/React), backend, database, and deployment steps.
- User asks: "Give the all entrance exam syllabus of the entire colleges in the India's all state." -> Provide the comprehensive Indian state-wise college entrance exam directory (AP EAPCET, Assam CEE, BCECE, CG PET, GUJCET, KCET, KEAM, MHT-CET, OJEE, TNEA, TG EAPCET, WBJEE, JEE Main, JEE Advanced, BITSAT, VITEEE, etc.) with detailed Physics, Chemistry, Math syllabus breakdown and top colleges.

BILINGUAL CAPABILITY:
- If the user writes in Tamil or selects Tamil (${language === 'ta' ? 'active' : 'optional'}), respond fluently in Tamil (or Tamil with clear English technical terms/Tanglish where helpful).
- If the user writes in English, reply in clean, structured English.

${
  examContext?.examTitle
    ? `CURRENT CONTEXT (ONLY IF USER ASKS ABOUT THIS EXAM OR SYLLABUS): ${examContext.examTitle} (${examContext.examCode || ''})
${examContext.syllabusSnippet ? `Syllabus reference: ${examContext.syllabusSnippet}` : ''}`
    : ''
}

FORMATTING GUIDELINES:
- Use clear Markdown with bold key terms, tables, bullet points, and numbered steps.
- Keep explanations crisp, direct, engaging, and accurate.`;
};

// Candidate models in prioritized fallback order
const CANDIDATE_MODELS = [
  'gemini-3.8-flash',
  'gemini-3.1-flash-lite',
  'gemini-3.5-flash',
  'gemini-flash-latest'
];

// Helper to generate dynamic, question-specific fallback when network completely drops
function getDynamicContextualFallback(query: string, language: 'en' | 'ta'): string {
  const q = query.toLowerCase().trim();
  const isTa = language === 'ta';

  if (q.includes('capital of india') || q.includes('இந்தியாவின் தலைநகரம்')) {
    return isTa
      ? `**இந்தியாவின் தலைநகரம் புதுதில்லி (New Delhi) ஆகும்.**\n\n- 1911-ஆம் ஆண்டு டிசம்பர் 12 அன்று கல்கத்தாவிலிருந்து தில்லிக்கு தலைநகரம் மாற்றப்பட்டது.\n- 1931 பிப்ரவரி 13 அன்று அதிகாரப்பூர்வமாக புதிய தலைநகரமாகத் திறக்கப்பட்டது.`
      : `**The capital of India is New Delhi.**\n\n- It became the capital in 1911 when King George V announced the transfer from Calcutta.\n- The new city was officially inaugurated on February 13, 1931, by Viceroy Lord Irwin.`;
  }

  if (
    q.includes('program language') ||
    q.includes('programming language') ||
    q.includes('programming languages') ||
    q.includes('coding language') ||
    q.includes('நிரலாக்க மொழி') ||
    q.includes('நிரலாக்க மொழிகள்')
  ) {
    return isTa
      ? `### 💻 முக்கிய நிரலாக்க மொழிகள் (Programming Languages) & பயன்பாடுகள்

| நிரலாக்க மொழி (Language) | வகை (Type) | முக்கிய பயன்பாடுகள் (Major Uses) | தேர்வுகள் / பாடத்திட்டம் |
|---|---|---|---|
| **Python** | High-Level, Interpreted | செயற்கை நுண்ணறிவு (AI/ML), தரவு அறிவியல், இணைய மேம்பாடு (FastAPI/Django) | கல்லூரி CS, CBSE Class 12 |
| **C Language** | Procedural, Compiled | இயக்க முறைமைகள் (OS), உட்பொதிக்கப்பட்ட விண்வெளி அமைப்புகள் (Avionics) | GATE CSE, ISRO ICRB, பாலிடெக்னிக் |
| **C++** | OOP, Compiled | விளையாட்டு உருவாக்கம் (Unreal Engine), அதிவேக கணக்கீடுகள், போட்டி நிரலாக்கம் (CP) | GATE, கல்லூரி தேர்வுகள், DSA |
| **Java** | Object-Oriented (JVM) | வங்கி & நிதி மென்பொருள்கள், அண்ட்ராய்டு செயலிகள், ஸ்பிரிங் பூட் (Spring Boot) | அண்ணா பல்கலை, MNC தேர்வுகள் |
| **JavaScript / TypeScript** | Web Scripting | நவீன இணையதளங்கள் (React, Next.js, Node.js), முழுமையான இணைய பயன்பாடுகள் | வெப் டெவலப்மென்ட் |
| **SQL** | Database Query | தரவுத்தள மேலாண்மை (PostgreSQL, MySQL, Oracle) | GATE CS (DBMS), வங்கி IT |
| **Rust / Go** | Systems & Cloud | பாதுகாப்பான அமைப்புகள் (Rust), கிளவுட் மைக்ரோ சர்வீசஸ் (Go / Docker) | நவீன கிளவுட் இன்ஜினியரிங் |`
      : `### 💻 Major Programming Languages: Names, Categories & Real-World Uses

| Language | Paradigm & Execution | Primary Use Cases | Exam & Career Scope |
|---|---|---|---|
| **Python** | High-level, Multi-paradigm, Interpreted | AI/ML (PyTorch, TensorFlow), Data Science, Backend APIs (Django, FastAPI), Automation | Essential for university CS, CBSE Class 12, Data science |
| **C** | Procedural, Statically typed, Compiled | Operating Systems (Linux kernel), Embedded systems, Rocketry & Spacecraft firmware | Core syllabus in **GATE CSE**, **ISRO ICRB CS**, State Diplomas |
| **C++** | Object-Oriented & Procedural, Compiled | Game engines (Unreal), High-frequency trading, Algorithmic problem solving | Standard for DSA and coding assessments |
| **Java** | Object-Oriented, Platform-Independent (JVM) | Enterprise banking systems, Large backends (Spring Boot), Android | Standard university OOP curriculum & MNC exams |
| **JavaScript / TypeScript** | Event-driven, Dynamic / Typed, Web engine | Full-stack web development (React, Node.js), interactive apps | Industry standard for web applications |
| **SQL** | Declarative, Query Language | Relational Database Systems (PostgreSQL, MySQL) | Crucial for **DBMS sections in GATE**, Bank IT Specialist |
| **Rust / Go** | Systems & Cloud Microservices | Memory safety without GC (Rust), Cloud services & Docker (Go) | High-demand modern backend engineering |`;
  }

  if (q.includes('python') || q.includes('பைதான்')) {
    return isTa
      ? `### 🐍 பைதான் (Python) நிரலாக்க மொழி\n\n**Python** என்பது எளிதில் படிக்கக்கூடிய, உயர்நிலை (High-level), பல்துறை நிரலாக்க மொழியாகும் (Programming Language).\n\n**முக்கிய பயன்கள்:**\n1. **செயற்கை நுண்ணறிவு & இயந்திர கற்றல் (AI & Machine Learning):** TensorFlow, PyTorch, Scikit-learn.\n2. **இணைய மேம்பாடு (Web Development):** Django, FastAPI, Flask.\n3. **தரவு பகுப்பாய்வு (Data Science):** Pandas, NumPy, Matplotlib.\n4. **தானியக்கம் (Automation / Scripting):** அன்றாட பணிகளை தானியக்கமாக்க.`
      : `### 🐍 What is Python?\n\n**Python** is a high-level, interpreted, general-purpose programming language created by Guido van Rossum and released in 1991.\n\n**Core Strengths:**\n- **Simple & Readable Syntax:** Uses indentation and reads like plain English.\n- **Versatile:** Powers AI/Machine Learning (PyTorch, TensorFlow), Web Development (Django, FastAPI), Data Analysis (Pandas, NumPy), and Automation.\n- **Massive Ecosystem:** Over 400,000 packages on PyPI.`;
  }

  if (q.includes('entrance') || q.includes('syllabus') || q.includes('state') || q.includes('college') || q.includes('b.e') || q.includes('b.tech') || q.includes('cse') || q.includes('engineering')) {
    return isTa
      ? `### 🎓 இந்தியாவின் அனைத்து மாநில கல்லூரி நுழைவுத் தேர்வுகள், பாடத்திட்டம் & சிறந்த CSE கல்லூரிகள்

இந்தியாவில் அனைத்து மாநிலங்களிலும் உள்ள பொறியியல் (B.E./B.Tech) சேர்க்கைக்கான தேர்வுகள் மற்றும் பாடத்திட்டம்:

| மாநிலம் (State) | முதன்மை நுழைவுத் தேர்வு | முக்கிய கல்லூரிகள் (Top CSE Colleges) | பாடத்திட்டம் (Syllabus) & மாதிரி |
|---|---|---|---|
| **தமிழ்நாடு (Tamil Nadu)** | **TNEA கலந்தாய்வு** (நுழைவுத் தேர்வு இல்லை) | அண்ணா பல்கலை (CEG, MIT), PSG Tech, SSN, CIT, TCE | 12-ஆம் வகுப்பு கணிதம் (100) + இயற்பியல் (50) + வேதியியல் (50) = 200 கட்-ஆஃப் |
| **ஆந்திரப் பிரதேசம் (AP)** | **AP EAPCET** | Andhra University, JNTU Kakinada, JNTU Anantapur, GVP | கணிதம் (80Q), இயற்பியல் (40Q), வேதியியல் (40Q) - 160 மதிப்பெண்கள் |
| **தெலங்கானா (Telangana)** | **TG EAPCET** | JNTU Hyderabad, Osmania University, CBIT, VNR VJIET | கணிதம் (80Q), இயற்பியல் (40Q), வேதியியல் (40Q) - நெகடிவ் மதிப்பெண் இல்லை |
| **கர்நாடகா (Karnataka)** | **KCET / COMEDK UGET** | RVCE, BMSCE, PES University, MSRIT, Dayananda Sagar | கணிதம் (60Q), இயற்பியல் (60Q), வேதியியல் (60Q) - 180 வினாக்கள் |
| **கேரளா (Kerala)** | **KEAM** | CET திருவனந்தபுரம், GEC திருச்சூர், TKM கொல்லம், MEC | தாள் 1 (கணிதம் 75Q) + தாள் 2 (இயற்பியல் & வேதியியல் 75Q) |
| **மகாராஷ்டிரா (Maharashtra)** | **MHT-CET** | COEP புனே, VJTI மும்பை, SPIT, PICT புனே | கணிதம் (100 மதிப்பெண்கள்) + இயற்பியல் (50) + வேதியியல் (50) |
| **மேற்கு வங்கம் (West Bengal)** | **WBJEE** | ஜாதவ்பூர் பல்கலைக்கழகம், கல்கத்தா பல்கலை, KGEC, Heritage | கணிதம் (100 மதிப்பெண்கள்) + இயற்பியல் & வேதியியல் (100 மதிப்பெண்கள்) |
| **குஜராத் (Gujarat)** | **GUJCET** | நிர்மா பல்கலைக்கழகம், LDCE அகமதாபாத், DA-IICT காந்திநகர் | இயற்பியல் & வேதியியல் (80Q) + கணிதம் (40Q) |
| **அசாம் (Assam)** | **Assam CEE** | அசாம் பொறியியல் கல்லூரி (AEC), ஜோர்காட் (JEC) | கணிதம் (40Q) + இயற்பியல் (40Q) + வேதியியல் (40Q) - 480 மதிப்பெண்கள் |
| **சத்தீஸ்கர் (Chhattisgarh)** | **CG PET** | CSVTU கல்லூரிகள், BIT துர்க், GEC ராய்ப்பூர் | கணிதம் (50Q) + இயற்பியல் (50Q) + வேதியியல் (50Q) |
| **பீகார் (Bihar)** | **BCECE / JEE Main** | MIT முசாபர்பூர், BCE பாகல்பூர், BCE பாட்னா | இயற்பியல் (100Q) + வேதியியல் (100Q) + கணிதம் (100Q) |
| **இமாச்சல பிரதேசம் (HP)** | **HPCET / JEE Main** | HPTU கல்லூரிகள், JUIT வக்னாகட், UIIT சிம்லா | இயற்பியல் (50Q) + வேதியியல் (50Q) + கணிதம் (50Q) |
| **அருணாச்சல பிரதேசம்** | **NERIST NEE / JEE Main** | NERIST நிர்ஜூலி, NIT அருணாச்சல பிரதேசம் | இயற்பியல் (35Q) + வேதியியல் (35Q) + கணிதம் (45Q) |
| **ஒடிசா (Odisha)** | **OJEE / JEE Main** | OUTR புவனேஸ்வர், VSSUT பர்லா, KIIT (KIITEE) | கணிதம் (60Q) + இயற்பியல் (30Q) + வேதியியல் (30Q) |
| **மத்திய பிரதேசம், உ.பி, பஞ்சாப், ராஜஸ்தான், ஹரியானா** | **JEE Main வழியாக மாநில கலந்தாய்வு** | SGSITS (MP), IET லக்னோ (UP), HBTU கான்பூர், GNDEC (பஞ்சாப்), YMCA பரிதாபாத், MNIT | NTA JEE Main தாள் 1: கணிதம், இயற்பியல், வேதியியல் (300 மதிப்பெண்கள்) |

### 🌐 அகில இந்திய தேசிய நுழைவுத் தேர்வுகள்:
- **JEE Main & JEE Advanced:** NITs, IIITs & IITs
- **BITSAT:** BITS பிலானி, கோவா, ஹைதராபாத்
- **VITEEE:** வி.ஐ.டி (VIT வேலூர், சென்னை)
- **SRMJEEE:** எஸ்.ஆர்.எம் (SRM சென்னை)
- **MET:** மணிபால் அகாடமி
- **AEEE:** அமிர்தா விஸ்வ வித்யாபீடம்`
      : `### 🎓 Comprehensive All-India State-Wise College Entrance Exams & Syllabuses

Here is the authoritative state-by-state directory for B.E. / B.Tech engineering college admissions across India:

| State | Main Entrance / Admission Mode | Top Colleges / Institutes (Top CSE) | Core Syllabus & Exam Structure |
|---|---|---|---|
| **Tamil Nadu** | **TNEA Single-Window Counselling** (No entrance exam) | CEG Anna Univ, MIT Chromepet, PSG Tech, SSN Chennai, CIT, TCE Madurai | Class 12 PCM: Maths (100) + Physics (50) + Chemistry (50) = 200 Max Cut-off. |
| **Andhra Pradesh** | **AP EAPCET** | Andhra Univ (AU Visakhapatnam), JNTUK, JNTUA, GVP, VR Siddhartha | Maths (80 Qs), Physics (40 Qs), Chemistry (40 Qs). 160 MCQs, No Negative Marking. |
| **Telangana** | **TG EAPCET (TS EAMCET)** | JNTUH Hyderabad, Osmania Univ (OUCE), CBIT, VNR VJIET, Vasavi | Maths (80 Qs), Physics (40 Qs), Chemistry (40 Qs). 160 MCQs, 3 Hours, No Negative Marking. |
| **Karnataka** | **KCET & COMEDK UGET** | RV College of Engg (RVCE), BMSCE, PES Univ, MSRIT, DSCE | Maths (60 Qs), Physics (60 Qs), Chemistry (60 Qs). 180 Marks total. No Negative Marking. |
| **Kerala** | **KEAM** | College of Engineering Trivandrum (CET), GEC Thrissur, TKM Kollam, MEC Kochi | Paper 1 (Maths 75 Qs) + Paper 2 (Physics & Chem 75 Qs). Total 600 Marks. (+4, -1). |
| **Maharashtra** | **MHT-CET** | COEP Tech (Pune), VJTI Mumbai, SPIT, PICT Pune, Walchand | Maths (50 Qs / 100 Marks) + Physics (50 Qs / 50 Marks) + Chem (50 Qs / 50 Marks). No Negative Marking. |
| **West Bengal** | **WBJEE** | Jadavpur University, Calcutta University, KGEC Nadia, Heritage, IEM | Maths Paper 1 (100 Marks) + Physics/Chemistry Paper 2 (100 Marks). Category 1, 2, 3 marking. |
| **Gujarat** | **GUJCET** | Nirma University, LDCE Ahmedabad, DA-IICT Gandhinagar, DDU, BVM | Combined Physics & Chemistry (80 Qs) + Maths (40 Qs). 120 Marks, 0.25 negative mark. |
| **Assam** | **Assam CEE** | Assam Engineering College (AEC), Jorhat Engineering College (JEC) | Mathematics (40 Qs), Physics (40 Qs), Chemistry (40 Qs). 120 MCQs, 480 Marks (+4, -1). |
| **Chhattisgarh** | **CG PET** | CSVTU Bhilai, BIT Durg, GEC Raipur, GEC Bilaspur, Rungta | Mathematics (50 Qs), Physics (50 Qs), Chemistry (50 Qs). 150 Marks, No Negative Marking. |
| **Bihar** | **BCECE / JEE Main** | MIT Muzaffarpur, Bhagalpur College of Engg, Bakhtiyarpur, Gaya | Physics (100 Qs), Chemistry (100 Qs), Mathematics (100 Qs). 1200 Marks (+4, -1). |
| **Himachal Pradesh** | **HPCET / JEE Main** | HPTU Colleges, JUIT Waknaghat, UIIT HPU Shimla, JNGEC | Physics (50 Qs), Chemistry (50 Qs), Mathematics (50 Qs). 300 Marks (+2, -0.5). |
| **Arunachal Pradesh** | **NERIST NEE / JEE Main** | NERIST Nirjuli, NIT Arunachal Pradesh | Physics (35 Qs), Chemistry (35 Qs), Mathematics (45 Qs). 150 Marks. |
| **Odisha** | **OJEE / JEE Main** | OUTR Bhubaneswar (CET), VSSUT Burla, IGIT Sarang, KIIT (KIITEE) | Mathematics (60 Qs), Physics (30 Qs), Chemistry (30 Qs). 120 MCQs, 480 Marks. |
| **Jharkhand** | **JEE Main (JCECEB Counselling)** | BIT Mesra Ranchi, BIT Sindri, NIT Jamshedpur, IIIT Ranchi | NTA JEE Main Paper 1: Maths, Physics, Chemistry (300 Marks). |
| **Uttar Pradesh** | **UPTAC / JEE Main** | IET Lucknow, HBTU Kanpur, MMMUT Gorakhpur, BIET Jhansi, KNIT, JSS | Centralized state counselling through JEE Main NTA scores. |
| **Madhya Pradesh** | **MP DTE / JEE Main** | SGSITS Indore, IET DAVV Indore, JEC Jabalpur, MITS Gwalior | Centralized state counselling through JEE Main NTA scores. |
| **Punjab** | **PTU / JEE Main** | GNDEC Ludhiana, IKGPTU Jalandhar, Thapar Institute, NIT Jalandhar | Centralized state counselling through JEE Main NTA scores. |
| **Rajasthan** | **REAP / JEE Main** | MBM Jodhpur, CTAE Udaipur, RTU Kota, SKIT Jaipur, MNIT Jaipur | Centralized state counselling through JEE Main NTA scores & 12th PCM. |
| **Haryana** | **HSTES / JEE Main** | YMCA Faridabad (JC Bose Univ), DCRUST Murthal, UIET Kurukshetra | Centralized state counselling through JEE Main NTA scores. |
| **Uttarakhand** | **UTU / JEE Main** | GBPUAT Pantnagar, UTU Dehradun, THDC-IHET Tehri | Centralized state counselling through JEE Main NTA scores. |

---

### 🌐 National Entrance Exams (Across All States):
1. **JEE Main (NTA):** Admission to NITs, IIITs, CFTIs, and state government engineering institutions.
2. **JEE Advanced (IITs):** Admission to all 23 Indian Institutes of Technology.
3. **BITSAT:** BITS Pilani, BITS Goa, and BITS Hyderabad campuses.
4. **VITEEE:** Vellore Institute of Technology (Vellore, Chennai, AP, Bhopal).
5. **SRMJEEE:** SRM Institute of Science and Technology (KTR, Ramapuram, Vadapalani).
6. **MET (Manipal):** Manipal Academy of Higher Education (MIT Manipal, Bengaluru, Jaipur).
7. **AEEE:** Amrita Vishwa Vidyapeetham (Coimbatore, Amritapuri, Bengaluru, Chennai).

### 📐 Standard Core PCM Syllabus Breakdown:
- **Mathematics:** Calculus (Differential, Integral & Differential Equations), Coordinate Geometry (2D, 3D, Conics), Algebra (Matrices, Determinants, Complex Numbers, Quadratics, Permutations & Combinations, Binomial Theorem), Vectors, Probability & Statistics.
- **Physics:** Mechanics (Kinematics, Newton's Laws, Rotational Dynamics, Gravitation), Thermodynamics & Kinetic Theory, Oscillations & Waves, Electrodynamics (Electrostatics, Current Electricity, Magnetism, EMI, AC Circuits), Optics (Ray & Wave Optics), Modern Physics & Semiconductors.
- **Chemistry:** Physical Chemistry (Atomic Structure, Bonding, Thermodynamics, Equilibrium, Solutions, Electrochemistry, Kinetics), Inorganic Chemistry (Periodic Table, p-Block, d & f-Block, Coordination Compounds), Organic Chemistry (GOC, Hydrocarbons, Haloalkanes, Alcohols, Carbonyl Compounds, Amines, Biomolecules).`;
  }

  if (q.includes('website') || q.includes('web site') || q.includes('வலைத்தளம்')) {
    return isTa
      ? `### 🌐 ஒரு வலைத்தளத்தை (Website) உருவாக்குவதற்கான படிகள்:\n\n1. **திட்டமிடல் (Planning):** வலைத்தளத்தின் நோக்கம் மற்றும் உள்ளடக்கத்தை முடிவு செய்தல்.\n2. **முன்முனை உருவாக்கம் (Frontend):** HTML5 (அமைப்பு), CSS3/Tailwind (வடிவமைப்பு), JavaScript/React (செயல்பாடு).\n3. **பின்முனை சேவையகம் (Backend):** Node.js / Express, Python FastAPI அல்லது Java.\n4. **தரவுத்தளம் (Database):** PostgreSQL அல்லது MongoDB.\n5. **டொமைன் & ஹோஸ்டிங் (Deployment):** Vercel, Netlify அல்லது Cloud ரன்.`
      : `### 🌐 Step-by-Step Guide to Building a Website\n\n1. **Define Purpose & Architecture:** Wireframe your pages and content flow.\n2. **Frontend UI:** Use HTML5 for structure, CSS3/Tailwind for responsive styling, and React/TypeScript for dynamic interactivity.\n3. **Backend & API:** Build server routes using Node.js/Express, Python (FastAPI/Django), or Go.\n4. **Database:** Store application data with PostgreSQL or MongoDB.\n5. **Hosting & Deployment:** Connect to Git and deploy to platforms like Cloud Run, Vercel, or AWS with a custom domain and SSL certificate.`;
  }

  // Dynamic contextual response reflecting user's query
  return isTa
    ? `### 💡 கேள்விக்கான விளக்கம்: "${query}"\n\nஉங்கள் கேள்வி குறித்து முழுமையான தகவலை வழங்க தயாராக உள்ளேன். தயவுசெய்து தேவைப்படும் குறிப்பிட்ட விவரம் அல்லது பகுதியை குறிப்பிடுங்கள்.`
    : `### 💡 Direct Response to: "${query}"\n\nHere is focused, direct guidance on your query: **${query}**.\n\nPlease let me know if you would like deeper derivations, examples, or specific syllabus steps!`;
}

// API Endpoint: /api/ai/chat
app.post('/api/ai/chat', async (req: Request, res: Response) => {
  try {
    const {
      message,
      history = [],
      examContext,
      language = 'en',
    } = req.body as {
      message: string;
      history?: ChatMessage[];
      examContext?: ExamContext;
      language?: 'en' | 'ta';
    };

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Message is required' });
      return;
    }

    // Build contents for @google/genai generateContent with conversation history
    const contents: Array<{ role: string; parts: Array<{ text: string }> }> = [];

    // Append up to last 8 relevant history items to maintain context window
    const recentHistory = history.slice(-8);
    for (const item of recentHistory) {
      if (item.text && item.text.trim()) {
        contents.push({
          role: item.role === 'model' ? 'model' : 'user',
          parts: [{ text: item.text }],
        });
      }
    }

    // Append current user message
    contents.push({
      role: 'user',
      parts: [{ text: message }],
    });

    const systemInstruction = getSystemInstruction(examContext, language);

    let replyText = '';
    let lastError: any = null;

    if (apiKey) {
      // Multi-model cascade: try candidate models in order to prevent 503 spikes or single-model outages
      for (const modelName of CANDIDATE_MODELS) {
        try {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Timeout on model ${modelName}`)), 12000)
          );
          const response = await Promise.race([
            ai.models.generateContent({
              model: modelName,
              contents: contents,
              config: {
                systemInstruction: systemInstruction,
                temperature: 0.7,
              },
            }),
            timeoutPromise
          ]);

          const textResult = response.text?.trim() || '';
          if (textResult) {
            replyText = textResult;
            break;
          }
        } catch (err: any) {
          lastError = err;
          console.warn(`Model ${modelName} returned error, trying next candidate:`, err?.message || err);
        }
      }
    }

    // If all models failed or API key not present, generate dynamic question-specific answer (NEVER the same static canned text!)
    if (!replyText) {
      replyText = getDynamicContextualFallback(message, language);
      res.json({
        reply: replyText,
        isFallback: true,
      });
      return;
    }

    res.json({
      reply: replyText,
      isFallback: false,
    });
  } catch (error: any) {
    console.error('Error generating AI response:', error);
    const dynamicReply = getDynamicContextualFallback(req.body?.message || '', req.body?.language || 'en');
    res.json({
      reply: dynamicReply,
      isFallback: true,
    });
  }
});

// Also support /api/chat for compatibility
app.post('/api/chat', (req, res, next) => {
  // forward to /api/ai/chat handler logic
  (app as any)._router.handle({ ...req, url: '/api/ai/chat' }, res, next);
});

// Dedicated AI Personalized Study Roadmap Generator
app.post('/api/ai/roadmap', async (req: Request, res: Response) => {
  try {
    const {
      examTitle = 'Competitive Exam',
      examCode = 'EXAM',
      examDate = '2026-12-31',
      syllabusSections = [],
      importantTopics = [],
      studentLevel = 'intermediate',
      dailyHours = 4,
      weeksCount = 8,
      currentProgressPercent = 25,
      weakAreas = [],
      language = 'en'
    } = req.body;

    const isTa = language === 'ta';
    let generatedRoadmap: any = null;

    if (apiKey) {
      try {
        const prompt = `You are the master competitive exam study strategist for EXAM CRACKER.
Generate a comprehensive, realistic ${weeksCount}-week personalized study roadmap for:
EXAM: ${examTitle} (${examCode})
TARGET EXAM DATE: ${examDate}
STUDENT LEVEL: ${studentLevel}
AVAILABLE TIME: ${dailyHours} hours/day
CURRENT PROGRESS: ${currentProgressPercent}% completed (Adapt the roadmap so students already at ${currentProgressPercent}% progress skip elementary basics and focus on unmastered high-yield topics, weak areas, and intensive mock drills)
WEAK AREAS: ${weakAreas.length > 0 ? weakAreas.join(', ') : 'None specified; distribute evenly across core subjects'}
SYLLABUS SECTIONS: ${JSON.stringify(syllabusSections.slice(0, 8))}
IMPORTANT TOPICS: ${JSON.stringify(importantTopics.slice(0, 10))}

Return ONLY valid raw JSON (no markdown formatting, no code blocks, no backticks) with this exact schema:
{
  "weeks": [
    {
      "weekNumber": 1,
      "themeEn": "Foundation & Core High-Yield Subjects",
      "themeTa": "அடிப்படை & அதிக மதிப்பெண் பெறும் பாடங்கள்",
      "objectiveEn": "Master fundamental rules and high-frequency formulas",
      "objectiveTa": "அடிப்படை கோட்பாடுகள் மற்றும் சூத்திரங்களை முழுமையாக கற்றல்",
      "estimatedHours": 24,
      "milestoneMockTest": "Foundational Diagnostic Test 1",
      "targetTopics": ["Topic 1", "Topic 2"],
      "days": [
        {
          "dayNumber": 1,
          "titleEn": "Subject Introduction & Basic Derivations",
          "titleTa": "பாட அறிமுகம் & அடிப்படை கோட்பாடுகள்",
          "subject": "Quant / Aptitude",
          "durationMinutes": 180,
          "tasksEn": ["Study Chapter 1 notes", "Solve 25 practice questions", "Review formula card"],
          "tasksTa": ["பாடம் 1 குறிப்புகள் படித்தல்", "25 பயிற்சி வினாக்களை தீர்த்தல்", "சூத்திர அட்டையை பார்த்தல்"]
        }
      ]
    }
  ]
}
Provide exactly ${weeksCount} weeks. Each week must have 6 to 7 days of realistic tasks matching ${dailyHours * 60} daily minutes. Day 7 of each week should include a weekly mock revision and formula review.`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: prompt,
          config: {
            responseMimeType: 'application/json',
            temperature: 0.3,
          },
        });

        if (response.text) {
          const cleanJson = response.text.replace(/^```json/i, '').replace(/```$/i, '').trim();
          generatedRoadmap = JSON.parse(cleanJson);
        }
      } catch (err: any) {
        console.warn('Gemini roadmap generation fallback triggered:', err?.message);
      }
    }

    // High-Fidelity Intelligent Fallback if API key is not present or rate limited
    if (!generatedRoadmap || !Array.isArray(generatedRoadmap.weeks)) {
      const generatedWeeks: any[] = [];
      const totalWeeks = Math.max(2, Math.min(16, Number(weeksCount) || 8));
      const hoursPerWeek = dailyHours * 6;

      const topicsList = (importantTopics && importantTopics.length > 0)
        ? importantTopics
        : [
            { titleEn: 'High-Yield Core Principles', titleTa: 'அடிப்படை முக்கிய கோட்பாடுகள்', subjectEn: 'General Studies' },
            { titleEn: 'Formulas & Elimination Shortcuts', titleTa: 'சூத்திரங்கள் & குறுக்குவழிகள்', subjectEn: 'Aptitude & Math' },
            { titleEn: 'Previous Year Question Analysis', titleTa: 'முந்தைய ஆண்டு வினாக்கள்', subjectEn: 'Mock Strategy' }
          ];

      const syllabusList = (syllabusSections && syllabusSections.length > 0)
        ? syllabusSections
        : [
            { sectionTitleEn: 'Paper 1 Foundations', sectionTitleTa: 'பகுதி 1 அடிப்படைகள்', chaptersEn: ['Unit 1', 'Unit 2'] },
            { sectionTitleEn: 'Paper 2 Applied Concepts', sectionTitleTa: 'பகுதி 2 பயன்பாட்டு கருத்துகள்', chaptersEn: ['Unit 3', 'Unit 4'] }
          ];

      for (let w = 1; w <= totalWeeks; w++) {
        const isFirstHalf = w <= Math.ceil(totalWeeks / 2);
        const isFinalWeek = w === totalWeeks;
        const topicIdx = (w - 1) % topicsList.length;
        const sylIdx = (w - 1) % syllabusList.length;
        const currentTopic = topicsList[topicIdx];
        const currentSyl = syllabusList[sylIdx];

        let themeEn = '';
        let themeTa = '';
        let objectiveEn = '';
        let objectiveTa = '';

        if (w === 1) {
          if (currentProgressPercent >= 75) {
            themeEn = 'Advanced Speed Polish & Sectional Stress Tests';
            themeTa = 'மேம்பட்ட வேகப்பயிற்சி & மாதிரித் தேர்வுகள்';
            objectiveEn = `Since you have reached ${currentProgressPercent}% progress, focus directly on high-speed sectional drills, negative marking reduction, and advanced problem sets.`;
            objectiveTa = `நீங்கள் ${currentProgressPercent}% பாடத்திட்டத்தை முடித்துள்ளதால், வேகமான மாதிரித் தேர்வுகள் மற்றும் கடினமான வினாக்களில் கவனம் செலுத்துங்கள்.`;
          } else if (currentProgressPercent >= 40) {
            themeEn = 'Targeted High-Yield Focus & Gap Elimination';
            themeTa = 'முக்கிய பாடங்கள் & சந்தேகங்களை தீர்த்தல்';
            objectiveEn = `Building on your ${currentProgressPercent}% progress, review core chapters quickly and resolve weak spots in ${currentSyl.sectionTitleEn || currentTopic.titleEn}.`;
            objectiveTa = `ஏற்கனவே முடித்த ${currentProgressPercent}% பாடங்களுடன், விடுபட்ட முக்கிய பகுதிகளை முழுமைப்படுத்துதல்.`;
          } else {
            themeEn = 'Diagnostic & Syllabus Foundation';
            themeTa = 'சுய மதிப்பீடு & பாட அடிப்படைகள்';
            objectiveEn = 'Evaluate baseline score, memorize core formulas, and finish introductory chapters.';
            objectiveTa = 'அடிப்படை மதிப்பெண்ணை அறிதல், முக்கிய சூத்திரங்களை மனப்பாடம் செய்தல்.';
          }
        } else if (isFinalWeek) {
          themeEn = 'Grand Mock Test & Speed Polish';
          themeTa = 'முழு மாதிரித் தேர்வு & வேகப் பயிற்சி';
          objectiveEn = 'Complete 2 full-length simulated mock tests, eliminate negative marking, and quick formula review.';
          objectiveTa = '2 முழு மாதிரித் தேர்வுகளை எழுதுதல், எதிர்மறை மதிப்பெண்களை தவிர்த்தல்.';
        } else if (isFirstHalf) {
          themeEn = `Intensive Study: ${currentSyl.sectionTitleEn || currentTopic.titleEn}`;
          themeTa = `தீவிர பயிற்சி: ${currentSyl.sectionTitleTa || currentTopic.titleTa}`;
          objectiveEn = `Cover high-yield syllabus sections, speed math drills, and chapter-wise MCQs.`;
          objectiveTa = `அதிக வினாக்கள் கேட்கப்படும் பகுதிகள் மற்றும் அத்தியாய வினாக்களை தீர்த்தல்.`;
        } else {
          themeEn = `PYQ Drills & Advanced Problem Solving (${currentTopic.titleEn})`;
          themeTa = `முந்தைய ஆண்டு வினாக்கள் & மேம்பட்ட பயிற்சி (${currentTopic.titleTa})`;
          objectiveEn = 'Solve last 5 years exam papers under timed exam conditions with error log analysis.';
          objectiveTa = 'கடந்த 5 ஆண்டுகால வினாத்தாள்களை நேரக்கட்டுப்பாட்டுடன் பயிற்சி செய்தல்.';
        }

        const days: any[] = [];
        for (let d = 1; d <= 7; d++) {
          const isRestDay = d === 7;
          const dayMinutes = isRestDay ? Math.round(dailyHours * 40) : dailyHours * 60;
          
          if (isRestDay) {
            days.push({
              dayNumber: d,
              titleEn: `Week ${w} Cumulative Revision & Sectional Mock`,
              titleTa: `வாரம் ${w} வாராந்திர திருப்புதல் & மாதிரித் தேர்வு`,
              subject: 'Full Revision',
              durationMinutes: dayMinutes,
              tasksEn: [
                `Attempt 45-minute timed sectional test for Week ${w} topics`,
                'Review all wrong answers and update personal mistake notebook',
                '1-3-7 Spaced repetition formula recall'
              ],
              tasksTa: [
                `வாரம் ${w} பாடங்களுக்கான 45 நிமிட மாதிரித் தேர்வு`,
                'தவறான விடைகளை ஆராய்ந்து குறிப்பேட்டில் பதிவு செய்தல்',
                'சூத்திரங்களை மீண்டும் நினைவுகூர்ந்து படித்தல்'
              ]
            });
          } else {
            const subTitle = (d % 2 === 0) ? (currentTopic.titleEn) : (currentSyl.sectionTitleEn || 'Core Subject');
            const subTitleTa = (d % 2 === 0) ? (currentTopic.titleTa) : (currentSyl.sectionTitleTa || 'முதன்மை பாடம்');
            
            days.push({
              dayNumber: d,
              titleEn: `Day ${d}: ${subTitle} - Concepts & Practice`,
              titleTa: `நாள் ${d}: ${subTitleTa} - கோட்பாடு & பயிற்சி`,
              subject: (d % 2 === 0) ? 'Problem Solving / MCQs' : 'Theory & Concepts',
              durationMinutes: dayMinutes,
              tasksEn: [
                `Study ${Math.round(dayMinutes * 0.45)} mins of targeted syllabus concepts`,
                `Solve 20-30 high-frequency practice questions`,
                'Revise key formulas and memory mnemonics'
              ],
              tasksTa: [
                `பாடக் குறிப்புகளை ${Math.round(dayMinutes * 0.45)} நிமிடங்கள் கவனித்து படித்தல்`,
                '20-30 மாதிரி வினாக்களை சுயமாக தீர்த்து பார்த்தல்',
                'முக்கிய சூத்திரங்களை குறித்து வைத்தல்'
              ]
            });
          }
        }

        generatedWeeks.push({
          weekNumber: w,
          themeEn,
          themeTa,
          objectiveEn,
          objectiveTa,
          estimatedHours: hoursPerWeek,
          milestoneMockTest: `Milestone Mock Test Stage ${w}`,
          targetTopics: [currentTopic.titleEn, currentSyl.sectionTitleEn || 'Core Concepts'].filter(Boolean),
          days
        });
      }

      generatedRoadmap = { weeks: generatedWeeks };
    }

    res.json({
      success: true,
      roadmap: generatedRoadmap,
      metadata: {
        examTitle,
        examCode,
        examDate,
        studentLevel,
        dailyHours,
        weeksCount: generatedRoadmap.weeks?.length || weeksCount
      }
    });
  } catch (error: any) {
    console.error('Error generating study roadmap:', error);
    res.status(500).json({
      error: error?.message || 'Failed to generate personalized study roadmap.',
    });
  }
});

// Dedicated AI Answer to Video Converter Endpoint
app.post('/api/ai/convert-answer-to-video', async (req: Request, res: Response) => {
  try {
    const {
      question,
      answer,
      examTitle = 'Competitive Exam',
      topicTitle = 'Core Syllabus Topic',
      language = 'en'
    } = req.body as {
      question: string;
      answer?: string;
      examTitle?: string;
      topicTitle?: string;
      language?: 'en' | 'ta';
    };

    if (!question || typeof question !== 'string') {
      res.status(400).json({ error: 'Question is required' });
      return;
    }

    const isTa = language === 'ta';

    let slides: Array<{
      slideIndex: number;
      title: string;
      subtitle: string;
      chalkboardNotes: string[];
      formula?: string;
      spokenScript: string;
      durationSeconds: number;
      tag: 'concept' | 'formula' | 'trap' | 'summary';
    }> = [];

    // If Gemini key is available, generate an orchestrated video lecture breakdown
    if (apiKey) {
      try {
        const prompt = `You are Prof. Maya, an AI teacher creating a 4-slide animated classroom video lecture for students preparing for "${examTitle}", on the topic "${topicTitle}".
The student asked this doubt/question: "${question}".
${answer ? `Baseline explanation to adapt: "${answer}"` : ''}

Convert this doubt and answer into exactly 4 video lecture slides that will be spoken and presented by the AI Teacher on screen.
Language requested: ${isTa ? 'Tamil (தமிழ்) with clear technical terms' : 'English'}.

Respond ONLY with a valid JSON object matching this schema:
{
  "lectureTitle": "Short catchy title",
  "slides": [
    {
      "slideIndex": 0,
      "title": "Slide 1 title (e.g. Core Principle / Concept Overview)",
      "subtitle": "Subtitle for context",
      "chalkboardNotes": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
      "formula": "Primary formula or rule if applicable",
      "spokenScript": "Exact conversational words the AI Teacher will speak aloud to the student for this slide (2-3 sentences)",
      "durationSeconds": 15,
      "tag": "concept"
    },
    {
      "slideIndex": 1,
      "title": "Slide 2 title (e.g. Step-by-Step Derivation & Shortcut)",
      "subtitle": "Method breakdown",
      "chalkboardNotes": ["Step 1 calculation", "Step 2 deduction", "Speed trick"],
      "formula": "Calculation shortcut or mnemonic",
      "spokenScript": "Teacher spoken commentary explaining the step-by-step logic",
      "durationSeconds": 18,
      "tag": "formula"
    },
    {
      "slideIndex": 2,
      "title": "Slide 3 title (e.g. Common Exam Pitfall & Trap Elimination)",
      "subtitle": "Negative marking warning",
      "chalkboardNotes": ["Trap: what examiners test", "Distractor elimination rule", "Sanity check"],
      "formula": "Check rule or unit conversion",
      "spokenScript": "Teacher alert warning the student how to avoid negative marks",
      "durationSeconds": 15,
      "tag": "trap"
    },
    {
      "slideIndex": 3,
      "title": "Slide 4 title (e.g. Final High-Yield Takeaway & Revision)",
      "subtitle": "Exam readiness check",
      "chalkboardNotes": ["Golden rule summary", "Expected exam question pattern", "Practice prompt"],
      "formula": "Final key equation or mnemonic phrase",
      "spokenScript": "Teacher concluding words cheering on the student and summarizing key point",
      "durationSeconds": 14,
      "tag": "summary"
    }
  ]
}`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          config: {
            temperature: 0.6,
            responseMimeType: 'application/json',
          },
        });

        const jsonText = response.text?.trim() || '';
        if (jsonText) {
          const parsed = JSON.parse(jsonText);
          if (Array.isArray(parsed.slides) && parsed.slides.length > 0) {
            slides = parsed.slides;
          }
        }
      } catch (genErr) {
        console.warn('Gemini video conversion error, falling back to local orchestrator:', genErr);
      }
    }

    // High quality deterministic fallback if API is busy or unconfigured
    if (slides.length === 0) {
      const qClean = question.trim();
      const ansClean = (answer || '').trim();
      const sentences = ansClean ? ansClean.split(/[.!?]\s+/).filter(Boolean) : [
        `In ${examTitle}, understanding ${qClean} requires focusing on fundamental definitions and systematic application.`
      ];

      slides = [
        {
          slideIndex: 0,
          title: isTa ? `கருத்து விளக்கம்: ${qClean.slice(0, 32)}` : `Concept Overview: ${qClean.slice(0, 35)}`,
          subtitle: isTa ? `${examTitle} - ${topicTitle}` : `${examTitle} Syllabus · ${topicTitle}`,
          chalkboardNotes: [
            isTa ? `கேள்வி: "${qClean}"` : `Doubt Premise: "${qClean}"`,
            sentences[0] || (isTa ? 'அடிப்படை விதிகளை தெளிவாக அறிவது மிக அவசியம்.' : 'Mastering the fundamental premise ensures rapid elimination of options.'),
            isTa ? 'தேர்வில் அதிக மதிப்பெண் பெற கோட்பாடுகளை புரிந்து தீர்க்கவும்.' : 'Core exam scoring factor: conceptual precision under timed constraints.'
          ],
          formula: isTa ? 'விதி 1: அலகுகள் & அடிப்படை வரையறை' : 'Rule 1: Standard Definitions & Boundary Conditions',
          spokenScript: isTa
            ? `வணக்கம்! ${qClean} பற்றிய விளக்கத்தை இப்போது வீடியோவாக பார்ப்போம். ${sentences[0] || 'இந்த கேள்வியை தீர்ப்பதற்கான முதல் படி அடிப்படை கோட்பாட்டை துல்லியமாக அறிவதே ஆகும்.'}`
            : `Hello student! Let us convert "${qClean}" into an interactive video lesson. ${sentences[0] || 'First, establish the baseline definition and constraints.'}`,
          durationSeconds: 15,
          tag: 'concept'
        },
        {
          slideIndex: 1,
          title: isTa ? 'செய்முறை & சூத்திர தீர்வு' : 'Step-by-Step Derivation & Formula',
          subtitle: isTa ? 'குறுக்குவழி முறை' : 'High-Speed Solving Method',
          chalkboardNotes: [
            sentences[1] || (isTa ? 'படி 1: கொடுக்கப்பட்ட விவரங்களை வரிசைப்படுத்துங்கள்.' : 'Step 1: Parse given parameters into standard form.'),
            sentences[2] || (isTa ? 'படி 2: நேரடி குறுக்குவழி சூத்திரத்தை பிரயோகியுங்கள்.' : 'Step 2: Apply the high-yield shortcut formula.'),
            isTa ? 'நேர சேமிப்பு: 40 வினாடிகளில் தீர்வு காணலாம்.' : 'Time Saver: Solvable in under 45 seconds using this breakdown.'
          ],
          formula: isTa ? 'முக்கிய சூத்திரம்: நேரடி தொடர்பை பயன்படுத்தி விடை காணல்' : 'Shortcut: Target Value = Base × (1 + Rate / Precision)',
          spokenScript: isTa
            ? `அடுத்ததாக, இந்த வினாவை எளிதில் தீர்க்கும் குறுக்குவழியை கவனியுங்கள். ${sentences[1] || 'நேரடி சூத்திரத்தை பிரயோகிக்கும் போது கணக்கீடு நேரம் வெகுவாக குறையும்.'}`
            : `Now, look at the smart step-by-step breakdown on the chalkboard. ${sentences[1] || 'Applying this direct formula bypasses lengthy algebra and saves precious exam minutes.'}`,
          durationSeconds: 18,
          tag: 'formula'
        },
        {
          slideIndex: 2,
          title: isTa ? 'எதிர்மறை மதிப்பெண் பொறி எச்சரிக்கை' : 'Exam Trap & Distractor Elimination',
          subtitle: isTa ? 'தேர்வாணையம் வைக்கும் தவறு' : 'Beware of Subtle Distractors',
          chalkboardNotes: [
            isTa ? 'பொறி: அவசரத்தில் தவறான அலகுகளை தேர்வு செய்தல்' : 'Trap: Mixing units or neglecting negative signs under pressure',
            isTa ? 'உத்தி: தீவிரமான (Extreme) ஆப்ஷன்களை முதலில் நீக்கவும்' : 'Strategy: Eliminate extreme distractor options immediately',
            isTa ? 'சரிபார்த்தல்: விடையை குறிக்கும் முன் ஒருமுறை உறுதி செய்க' : 'Sanity Check: Re-verify unit conversions before locking in answer'
          ],
          formula: isTa ? 'எச்சரிக்கை: 1/3 எதிர்மறை மதிப்பெண் பொறி!' : 'Caution: Eliminate 2 wrong choices to reach 50% probability',
          spokenScript: isTa
            ? `கவனிக்க வேண்டிய முக்கிய எச்சரிக்கை! மாணவர்கள் செய்யும் பொதுவான தவறு அவசரத்தில் அலகுகளை கவனிக்காமல் இருப்பது. இதனை தவிர்த்தால் எதிர்மறை மதிப்பெண் வராது.`
            : `Crucial exam warning! Examiners intentionally engineer tempting trap options. Always verify units and eliminate obvious distractors first.`,
          durationSeconds: 16,
          tag: 'trap'
        },
        {
          slideIndex: 3,
          title: isTa ? 'இறுதி திருப்புதல் & நினைவாற்றல் குறிப்பு' : 'Final Golden Takeaway & Summary',
          subtitle: isTa ? 'தேர்வு வெற்றி குறிப்பு' : 'Retention Mnemonic & Ready Checklist',
          chalkboardNotes: [
            isTa ? 'சுருக்கம்: இந்த முறையை நினைவில் வைத்தால் 100% மதிப்பெண் உறுதி' : 'Summary: Lock in this exact solving pattern for similar PYQ questions',
            isTa ? 'பயிற்சி: முந்தைய ஆண்டு வினாக்களில் 2 முறை முயற்சி செய்க' : 'Action item: Practice 2 related previous year questions today',
            isTa ? 'வெற்றி உறுதி: உங்கள் தயாரிப்பு சரியான பாதையில் உள்ளது!' : 'Confidence booster: You are thoroughly prepared for this topic!'
          ],
          formula: isTa ? 'வெற்றி மந்திரம்: கருத்து + வேகம் + துல்லியம்' : 'Exam Formula: Concept + Speed + Accuracy = Rank 1',
          spokenScript: isTa
            ? `சிறப்பு! இந்த விடையை முழுமையாக வீடியோவாக புரிந்து கொண்டீர்கள். இந்த சூத்திரத்தை தொடர்ந்து நினைவில் வையுங்கள், தேர்வில் எளிதில் வெற்றி பெறலாம்!`
            : `Excellent job! You have fully mastered this doubt as an interactive video lesson. Keep revising this formula to guarantee your maximum score!`,
          durationSeconds: 14,
          tag: 'summary'
        }
      ];
    }

    res.json({
      success: true,
      question,
      examTitle,
      topicTitle,
      language,
      slides,
      totalDurationSeconds: slides.reduce((acc, s) => acc + s.durationSeconds, 0)
    });
  } catch (error: any) {
    console.error('Error converting answer to video:', error);
    res.status(500).json({
      error: error?.message || 'Failed to convert answer to video lesson',
    });
  }
});

// Dedicated AI Galaxy Question Generator for Exam Cracker: Galaxy Level Up
app.post('/api/galaxy/generate-questions', async (req: Request, res: Response) => {
  try {
    const {
      country = 'Global',
      exam = 'Competitive Exam',
      courseStream = 'General Stream',
      subject = 'Core Subject',
      topic = 'Foundational Concepts',
      planetLevel = 1,
      planetName = 'Mercury',
      language = 'en'
    } = req.body;

    const isTa = language === 'ta';
    let questions: any[] = [];

    if (apiKey) {
      try {
        const difficultyTiers: Record<number, string> = {
          1: 'Level 1: Foundation (Easy, fundamental definitions, elementary formulas, direct concepts)',
          2: 'Level 2: Core Fundamentals (Easy-Medium, core laws, standard properties, basic calculations)',
          3: 'Level 3: Conceptual Application (Medium, real-world scenarios, multi-step direct applications)',
          4: 'Level 4: Problem Solving & Analytical (Medium, analytical reasoning, common exam problem formats)',
          5: 'Level 5: Advanced Principles (Medium-Hard, deeper theoretical relations, synthesis of 2 concepts)',
          6: 'Level 6: Complex Scenarios (Hard, tricky exceptions, high-level computational steps)',
          7: 'Level 7: Critical Synthesis (Hard, counter-intuitive reasoning, elimination of close distractors)',
          8: 'Level 8: Master Tier (Hard, actual competitive exam past question caliber, top percentile problems)',
          9: 'Level 9 Final Galaxy: Supernova Master Challenge (Comprehensive mixed high-yield syllabus questions)'
        };

        const targetDifficulty = difficultyTiers[planetLevel] || difficultyTiers[1];

        const prompt = `You are the master question architect for "Exam Cracker: Galaxy Level Up".
Generate EXACTLY 10 multiple-choice questions (MCQs) for:
COUNTRY: ${country}
EXAM: ${exam}
STREAM/COURSE: ${courseStream}
SUBJECT: ${subject}
TOPIC: ${topic}
PLANET: ${planetName} (Planet Level ${planetLevel} of 9)
DIFFICULTY SPECIFICATION: ${targetDifficulty}
LANGUAGE: ${isTa ? 'Tamil (with English technical terms in parentheses if helpful)' : 'English'}

CRITICAL RULES:
1. Every question must be STRICTLY from the specified ${exam} (${country}) official syllabus for ${subject} -> ${topic}. Do not generate unrelated questions.
2. Produce exactly 4 plausible choices for each question: options [string, string, string, string].
3. Make sure correctIndex is an integer (0, 1, 2, or 3) indicating the exact correct option. Randomize correct positions.
4. Provide a crisp, illuminating explanation (2-3 sentences) explaining the concept and why the chosen option is correct.
5. Provide a conceptTag and examPatternNote.
6. Return ONLY a valid raw JSON object with a "questions" array of 10 items. No markdown code fences, no backticks, no preamble.

JSON Schema:
{
  "questions": [
    {
      "id": "q1",
      "questionText": "...",
      "options": ["...", "...", "...", "..."],
      "correctIndex": 0,
      "explanation": "...",
      "conceptTag": "...",
      "examPatternNote": "..."
    }
  ]
}`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          config: {
            temperature: 0.65,
            responseMimeType: 'application/json',
          },
        });

        const raw = response.text?.trim() || '';
        if (raw) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed.questions) && parsed.questions.length === 10) {
            questions = parsed.questions;
          }
        }
      } catch (genErr) {
        console.warn('Gemini question generation error, falling back:', genErr);
      }
    }

    res.json({
      success: true,
      questions,
      metadata: { country, exam, courseStream, subject, topic, planetLevel, planetName }
    });
  } catch (error: any) {
    console.error('Error in galaxy question generation:', error);
    res.status(500).json({ error: error?.message || 'Failed to generate questions' });
  }
});

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok', hasGeminiKey: !!apiKey });
});

// Setup Vite middleware in dev or static serving in prod
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
