ภาพรวม

  Doc v2 ครอบคลุม organizational structure ของบริษัทขนาดเล็ก (10-50 คน) ได้ดี แต่ยังขาดกลไกสำคัญที่ทำให้ "เหมือน company จริง" — โดยเฉพาะ goal cascade, information classification, identity lifecycle, และ project layer. โครงสร้างปัจจุบันเหมือน startup flat
  hierarchy มากกว่า scaled organization

  ---
  P0 — ขาดแกนหลักของบริษัทจริง

  1. Department / Division Layer (4-tier instead of 3-tier)

  ปัจจุบัน Company → Team → Role โดดข้ามชั้นไป real companies มี:
  Company → Department/Division → Team → Sub-team → Role
  เหตุผล: Engineering Dept มี 5 teams, Sales Dept มี 3 teams — knowledge ของ Engineering Dept (engineering practices, tech stack standards) ไม่ใช่ของ company ทั้งหมด แต่ก็ไม่ใช่ของ team เดียว เพิ่ม vaults/departments/<dept>/ เป็น tier 1.5

  2. OKR / Goal Cascade (กลไก alignment ที่ขาดหายไป)

  ตอนนี้ KPI weights เป็น role-static — ไม่มีการ cascade เป้าหมายจาก company → team → individual. ต้องเพิ่ม:
  - company_objectives → quarterly OKRs
  - team_objectives → derived from company OKRs
  - individual_goals → derived from team OKRs
  - KPI weights ปรับ dynamic ตาม current quarter's OKR (ไม่ใช่ role-fixed)

  นี่คือกลไกที่ทำให้ "ทุกคนทำงานร่วมทิศทางเดียวกัน" ในบริษัทจริง

  3. Information Classification (ขาด sensitivity layer)

  "Company LTM อ่านได้ทุกคน" ผิดในบริษัทจริง — ต้องเพิ่ม classification:

  ┌──────────────┬─────────────────────────┬───────────────────────────────────────┐
  │    Level     │        Audience         │                Example                │
  ├──────────────┼─────────────────────────┼───────────────────────────────────────┤
  │ public       │ ทุกคน + external         │ Brand voice, public docs              │
  ├──────────────┼─────────────────────────┼───────────────────────────────────────┤
  │ internal     │ All members             │ Coding standards, OKRs                │
  ├──────────────┼─────────────────────────┼───────────────────────────────────────┤
  │ confidential │ Need-to-know            │ HR notes, customer data, compensation │
  ├──────────────┼─────────────────────────┼───────────────────────────────────────┤
  │ restricted   │ Specific role/clearance │ Trade secrets, security keys, legal   │
  └──────────────┴─────────────────────────┴───────────────────────────────────────┘

  ต้องเพิ่ม classification field ใน vault notes + access control matrix

  4. Identity & Membership Lifecycle

  ปัจจุบันเป็น static membership ขาด:
  - Onboarding: new member → reading list + buddy assignment
  - Transfer: agent ย้ายทีม → carry-over knowledge rules
  - Alumni: former member อ่านได้แต่ไม่เขียน (read-only legacy)
  - Multi-team membership: agent คนเดียวอยู่ 2 teams (matrix org)

  ---
  P1 — เพิ่มเพื่อ scale

  5. Project / Initiative Layer (time-bounded knowledge)

  Teams persistent แต่ projects time-bounded — ขาดมิตินี้:
  Project: { id, name, sponsor_team, member_teams[], start, end, status }
  Project LTM → archived เมื่อ project close
  ตัวอย่าง: "Hermes V2 Launch" project ข้าม dev-squad + content-studio → project vault

  6. Decision Provenance Metadata (formal ADR fields)

  ปัจจุบัน decisions เป็น .md ธรรมดา ต้องเพิ่ม:
  decided_by: <agent_id>
  decided_at: <timestamp>
  consulted: [<agent_id>, ...]
  inputs_considered: [...]
  review_by: <date>          # เมื่อต้อง re-evaluate
  supersedes: <prev_decision_id>
  status: proposed | accepted | superseded | deprecated

  7. Knowledge Ownership / Stewardship

  ทุก note ต้องมี owner ที่รับผิดชอบ — ไม่งั้น vault rot. เพิ่ม:
  - owner: <agent_id or team_id>
  - last_reviewed_at
  - review_cadence: 30d | 90d | 180d | annual
  - Auto-flag stale notes ใน /memory lint

  8. External Entities Layer

  บริษัทจริงไม่ได้อยู่ในสุญญากาศ ต้องมี:
  - entities/customers/ — customer accounts, feedback, contracts
  - entities/vendors/ — supplier knowledge
  - entities/partners/ — partnership context
  - entities/competitors/ — competitive intelligence (restricted classification)

  ---
  P2 — Real-world fidelity

  9. Communication Channel Types

  แต่ละ channel สร้าง knowledge รูปแบบต่างกัน:
  - 1-on-1 notes (private: manager + report เท่านั้น)
  - Meeting notes (team-scoped)
  - All-hands (company-wide)
  - Async threads (lower-fidelity)
  - ADRs (high-fidelity, formal)

  ปัจจุบัน doc ปฏิบัติทุกอย่างเหมือนกันหมด

  10. Reporting Lines + Performance Notes

  Lead/Worker เป็น depth-based แต่ขาด explicit manager-report graph:
  - Performance feedback (private to manager + report)
  - Career development goals
  - Mentorship pairings

  11. Audit Trail

  Compliance ต้องมี immutable log ของ who-did-what-when ใน company vault — ตอนนี้ใช้ updated_at เฉยๆ ไม่ tamper-proof

  12. Calibration / Human-in-loop Promotion

  Auto-promotion (2+ references) เป็น mechanical ในบริษัทจริงต้องมี calibration meeting — Lead review ก่อน promote สูงๆ. Doc มี suggestion system แต่ใช้แค่กับ non-leads ควรขยายให้ครอบคลุม automatic promotions ระดับ team→company ด้วย

  ---
  คำแนะนำลำดับความสำคัญ

  ถ้าจะปรับ doc:
  1. เพิ่ม section ใหม่: "§17 Information Classification" และ "§18 Identity Lifecycle" — สองอันนี้กระทบทุก section เดิม (access control, write rules, link resolution)
  2. ปรับ §3.1: เพิ่ม vaults/departments/ และ vaults/projects/
  3. ปรับ §6 (XP trigger): เพิ่ม OKR alignment ใน KPI weight calculation
  4. ปรับ §3.2 schema: เพิ่ม classification, owner, review_by, decided_by fields ใน vault note tables
  5. เพิ่ม §19: Project lifecycle (active → archived) + project vault tier

  โดยรวม v2 ดีพื้นฐานแล้ว แต่ feel เหมือน "team-based startup" มากกว่า "scaled company" — เติม 4 มิติ (Department, OKR, Classification, Lifecycle) จะกลายเป็น real-world fidelity ทันที