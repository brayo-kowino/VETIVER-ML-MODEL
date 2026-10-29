"""
Generates a professional Microsoft Word (.docx) document for the
VETIVER Team Structure & Operational Charter.
Uses standard Python library (zipfile, xml) with zero external dependencies.
"""
import os
import zipfile
from xml.sax.saxutils import escape


def create_docx(filename: str):
    # XML Components
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:color w:val="1E293B"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="260" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="52"/>
      <w:color w:val="1B5E20"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:i/>
      <w:sz w:val="26"/>
      <w:color w:val="475569"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:after="300"/>
    </w:pPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="34"/>
      <w:color w:val="2E7D32"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="360" w:after="140"/>
    </w:pPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:color w:val="1E293B"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="240" w:after="100"/>
    </w:pPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="22"/>
      <w:color w:val="334155"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="160" w:after="60"/>
    </w:pPr>
  </w:style>
</w:styles>"""

    # Builder helpers
    def p(text="", bold=False, italic=False, color=None, size=None, align=None, style=None):
        p_pr = []
        if style:
            p_pr.append(f'<w:pStyle w:val="{style}"/>')
        if align:
            p_pr.append(f'<w:jc w:val="{align}"/>')
        p_pr_xml = f"<w:pPr>{''.join(p_pr)}</w:pPr>" if p_pr else ""

        if not text:
            return f"<w:p>{p_pr_xml}</w:p>"

        r_pr = []
        if bold:
            r_pr.append("<w:b/>")
        if italic:
            r_pr.append("<w:i/>")
        if color:
            r_pr.append(f'<w:color w:val="{color}"/>')
        if size:
            r_pr.append(f'<w:sz w:val="{size}"/>')
        r_pr_xml = f"<w:rPr>{''.join(r_pr)}</w:rPr>" if r_pr else ""

        return f"<w:p>{p_pr_xml}<w:r>{r_pr_xml}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"

    def bullet(text, bold_prefix="", level=0):
        indent_val = 360 * (level + 1)
        r_prefix = f"<w:r><w:rPr><w:b/><w:color w:val=\"1E293B\"/></w:rPr><w:t xml:space=\"preserve\">{escape(bold_prefix)} </w:t></w:r>" if bold_prefix else ""
        return f"""<w:p>
          <w:pPr>
            <w:ind w:left="{indent_val}" w:hanging="240"/>
            <w:spacing w:after="80"/>
          </w:pPr>
          <w:r><w:t xml:space="preserve">&#8226; </w:t></w:r>
          {r_prefix}
          <w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>
        </w:p>"""

    def table(rows_data, col_widths=None):
        out = [
            '<w:tbl>',
            '<w:tblPr>',
            '<w:tblW w:w="5000" w:type="pct"/>',
            '<w:tblBorders>',
            '<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>',
            '<w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>',
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="94A3B8"/>',
            '<w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>',
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>',
            '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>',
            '</w:tblBorders>',
            '</w:tblPr>'
        ]
        for row_idx, row in enumerate(rows_data):
            out.append('<w:tr>')
            is_header = (row_idx == 0)
            for cell_idx, cell_text in enumerate(row):
                bg_color = "E8F5E9" if is_header else ("F8FAFB" if row_idx % 2 == 1 else "FFFFFF")
                bold_attr = '<w:b/>' if is_header else ''
                text_color = '1B5E20' if is_header else '1E293B'
                out.append(f"""<w:tc>
                  <w:tcPr>
                    <w:shd w:val="clear" w:color="auto" w:fill="{bg_color}"/>
                    <w:tcMar>
                      <w:top w:w="120" w:type="dxa"/>
                      <w:bottom w:w="120" w:type="dxa"/>
                      <w:left w:w="140" w:type="dxa"/>
                      <w:right w:w="140" w:type="dxa"/>
                    </w:tcMar>
                  </w:tcPr>
                  <w:p>
                    <w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
                    <w:r>
                      <w:rPr>{bold_attr}<w:sz w:val="19"/><w:color w:val="{text_color}"/></w:rPr>
                      <w:t xml:space="preserve">{escape(str(cell_text))}</w:t>
                    </w:r>
                  </w:p>
                </w:tc>""")
            out.append('</w:tr>')
        out.append('</w:tbl>')
        return "".join(out)

    def callout(title, body):
        return f"""<w:p>
          <w:pPr>
            <w:pBdr>
              <w:left w:val="single" w:sz="24" w:space="15" w:color="2E7D32"/>
            </w:pBdr>
            <w:shd w:val="clear" w:color="auto" w:fill="F4F7F5"/>
            <w:ind w:left="240" w:right="240"/>
            <w:spacing w:before="160" w:after="160"/>
          </w:pPr>
          <w:r><w:rPr><w:b/><w:color w:val="1B5E20"/></w:rPr><w:t xml:space="preserve">{escape(title)}: </w:t></w:r>
          <w:r><w:t xml:space="preserve">{escape(body)}</w:t></w:r>
        </w:p>"""

    # Assemble Document Content
    body = []

    # Title & Metadata
    body.append(p("VETIVER PROJECT — TEAM CHARTER & OPERATIONAL PLAN", style="Title"))
    body.append(p("AI Bio-Engineering Decision Support & Geotechnical Research Platform", style="Subtitle"))
    body.append(p("Institution: Jomo Kenyatta University of Agriculture and Technology (JKUAT)", bold=True, color="475569"))
    body.append(p("Author & Project Lead: Brian (Overall Project Leader & Principal Technical Lead)", bold=True, color="2E7D32"))
    body.append(p("Team Size: 6 Members  |  Deployment Target: JKUAT Workstations & WhatsApp Cloud API", italic=True, color="64748B"))
    body.append(p(""))

    # Section 1: Executive Summary
    body.append(p("1. Executive Summary & Team Structure", style="Heading1"))
    body.append(p("This document formalizes the organizational hierarchy, task ownership, GitHub collaboration workflows, and deployment roadmap for the 6-person VETIVER engineering team at JKUAT."))
    body.append(p("The team is structured into two core branches of three members each, operating under the overall leadership of the Technical Lead (Brian):"))
    body.append(bullet("Technical & Systems Division (3 Members): Responsible for core ML/RAG pipelines, WhatsApp API integration, FastAPI webhooks, Docker orchestration, and on-premise JKUAT workstation hosting.", bold_prefix="1."))
    body.append(bullet("Data & Sanitization Division (3 Members): Responsible for Google Drive dataset mining, schema enforcement, geotechnical unit standardization, peer-reviewed literature acquisition, and agronomic boundary validation.", bold_prefix="2."))

    # Organizational Matrix Table
    body.append(p(""))
    body.append(p("Table 1.1: 6-Member Team Topology & Leadership Matrix", style="Heading2"))
    top_matrix = [
        ["Role ID", "Member / Title", "Division", "Primary Focus Area", "Reports To"],
        ["Role 1", "Brian (Overall Project & Tech Lead)", "Technical", "Architecture, ML/RAG Integration, PR Review, Deployment", "Project Sponsor / Patron"],
        ["Role 2", "WhatsApp API Integration Engineer", "Technical", "Meta Cloud / Twilio API, FastAPI Webhooks, Mobile Formatting", "Brian"],
        ["Role 3", "Platform & MLOps Engineer", "Technical", "FastAPI Service, Docker Compose, JKUAT Desktop Setup, CI/CD", "Brian"],
        ["Role 4", "Data Lead & Quality Assurance Analyst", "Data & Sanitization", "Google Drive Master Sheet, Schema Hygiene, Unit Normalization", "Brian"],
        ["Role 5", "Literature & Document Sourcing Specialist", "Data & Sanitization", "PDF Acquisition (East African soils), Text Cleaning, Citations", "Data Lead (Role 4)"],
        ["Role 6", "Agronomic & Geotechnical Specialist", "Data & Sanitization", "Failure Boundaries, Field Physics, Model Ground-Truthing", "Data Lead (Role 4)"],
    ]
    body.append(table(top_matrix))
    body.append(p(""))

    # Section 2: Detailed Role Responsibilities & Backlogs
    body.append(p("2. Role Responsibilities & Actionable Task Backlogs", style="Heading1"))

    # Role 1
    body.append(p("Role 1: Overall Project Lead & Principal AI/ML Architect (Brian)", style="Heading2"))
    body.append(p("Direct Reports: WhatsApp Integration Engineer, Platform Engineer, Data Lead."))
    body.append(bullet("Maintain end-to-end vision, technical architecture, and codebase hygiene on GitHub.", "Mandate:"))
    body.append(bullet("Perform final code review and merge sign-off on all Pull Requests into the main branch.", "Mandate:"))
    body.append(bullet("Supervise the CatBoost predictive model and LlamaIndex vector store embeddings.", "Mandate:"))
    body.append(bullet("Coordinate joint physical sessions and lead the final JKUAT on-premise hardware deployment.", "Mandate:"))
    body.append(bullet("Audit weekly data batches from the Data Team before pushing into model retraining.", "Sprint Backlog:"))
    body.append(bullet("Review and test WhatsApp webhook integration endpoints created by Role 2.", "Sprint Backlog:"))
    body.append(bullet("Lead milestone review presentations with university patrons and external faculty reviewers.", "Sprint Backlog:"))

    # Role 2
    body.append(p("Role 2: WhatsApp API & Messaging Integration Engineer (Technical Team)", style="Heading2"))
    body.append(p("Direct Report To: Brian (Technical Lead)"))
    body.append(p("Field bio-engineers, farmers, and reviewers will interact with the platform directly via WhatsApp. This role is dedicated to the messaging bridge and integration."))
    body.append(bullet("Set up and manage the Meta WhatsApp Business Cloud API (or Twilio Sandbox).", "Key Task:"))
    body.append(bullet("Build whatsapp_service.py using FastAPI to listen for incoming webhooks with HMAC security.", "Key Task:"))
    body.append(bullet("Implement query routing logic: if the user sends comma-separated site parameters, route to CatBoost predictor; if the user asks a natural language question, route to LlamaIndex RAG.", "Key Task:"))
    body.append(bullet("Format outgoing responses for WhatsApp: chunk text at 1,000 characters, use clean bold headings, and append verified citations at the end.", "Key Task:"))
    body.append(bullet("Maintain user session states across multi-turn site assessment questionnaires.", "Key Task:"))

    # Role 3
    body.append(p("Role 3: Platform & MLOps Engineer (Technical Team)", style="Heading2"))
    body.append(p("Direct Report To: Brian (Technical Lead)"))
    body.append(p("Ensures high availability, local desktop optimization, and container management."))
    body.append(bullet("Maintain and test the Streamlit web interface (UI reactivity, real icons, fixed bottom bar).", "Key Task:"))
    body.append(bullet("Wrap core ML and RAG functions in a lightweight FastAPI application (api.py) exposing /predict, /chat, and /health endpoints.", "Key Task:"))
    body.append(bullet("Configure the JKUAT Desktop Workstation environment: install Ollama as a Windows/Linux persistent background service (NSSM / systemd) that boots automatically.", "Key Task:"))
    body.append(bullet("Manage Dockerfile and docker-compose.yml, ensuring persistent mounts for /storage, /data, and /knowledge_base.", "Key Task:"))
    body.append(bullet("Set up Cloudflare Tunnel (cloudflared) on the JKUAT machine to expose the WhatsApp webhook securely over HTTPS.", "Key Task:"))

    # Role 4
    body.append(p("Role 4: Data Lead & Quality Assurance Analyst (Data & Sanitization Team)", style="Heading2"))
    body.append(p("Direct Report To: Brian (Project Lead)"))
    body.append(p("Owns the empirical data pipeline from the shared Google Drive spreadsheet to the production dataset."))
    body.append(bullet("Supervise the online Google Drive / Google Sheets master workbook.", "Key Task:"))
    body.append(bullet("Enforce controlled vocabularies: eliminate spelling mismatches (e.g. 'Semi- arid.' vs 'semi-arid').", "Key Task:"))
    body.append(bullet("Standardize all geotechnical units: root length to cm, soil density, salinity to ECe (dS/m) or normalized categories.", "Key Task:"))
    body.append(bullet("Export cleaned, verified batches to data/vetiver_growth_data.csv and knowledge_base/vetiver_growth_data.csv.", "Key Task:"))
    body.append(bullet("Perform data leakage audits to ensure validation rows are never contaminated into training splits.", "Key Task:"))

    # Role 5
    body.append(p("Role 5: Literature & Document Sourcing Specialist (Data & Sanitization Team)", style="Heading2"))
    body.append(p("Direct Report To: Data Lead (Role 4)"))
    body.append(p("Expands the RAG knowledge base beyond tabular CSVs to include scientific papers, technical reports, and regional case studies."))
    body.append(bullet("Source 15-25 high-impact papers from TVNI (The Vetiver Network International), ICRAF, FAO, and geotechnical journals.", "Key Task:"))
    body.append(bullet("Target Kenyan and East African soil conditions: Central Kenya red volcanic soils, Rift Valley volcanic ash, and coastal sands.", "Key Task:"))
    body.append(bullet("Clean extracted PDF text: strip headers, page numbers, and reference lists, saving clean markdown files into knowledge_base/.", "Key Task:"))
    body.append(bullet("Tag each document with metadata: Author, Year, Region, and Key Topic for source attribution.", "Key Task:"))

    # Role 6
    body.append(p("Role 6: Agronomic & Geotechnical Domain Specialist (Data & Sanitization Team)", style="Heading2"))
    body.append(p("Direct Report To: Data Lead (Role 4)"))
    body.append(p("Serves as the scientific gatekeeper ensuring model predictions and chatbot answers reflect real soil mechanics."))
    body.append(bullet("Design realistic failure boundary cases: expand failure scenarios from 16 to 40+ rows (e.g., bedrock refusal, frost crown death, toxic leachate without neutralization).", "Key Task:"))
    body.append(bullet("Validate CatBoost TreeSHAP impact values: confirm that negative and positive feature impacts match geotechnical physics.", "Key Task:"))
    body.append(bullet("Formulate a 30-question 'Golden Evaluation Benchmark' covering slope stability, root shear strength, and flood regimes.", "Key Task:"))
    body.append(bullet("Audit chatbot responses during team review sessions to catch and eliminate factual hallucinations.", "Key Task:"))

    # Section 3: Governance & Cadence
    body.append(p(""))
    body.append(p("3. Governance & Communication Cadence", style="Heading1"))
    body.append(callout("Communication Protocol", "All asynchronous updates, daily blockers, and pull request alerts are communicated via the project WhatsApp group and GitHub Issues. Physical synchronization sessions are reserved for code walkthroughs and on-premise hardware deployments."))

    cadence_table = [
        ["Session Type", "Frequency", "Venue", "Primary Objective", "Mandatory Attendees"],
        ["Async Standup", "Daily (Mon-Fri)", "WhatsApp Group", "Report: 1. Completed, 2. Today's goal, 3. Blockers", "All 6 Members"],
        ["Weekly Tech Review", "Weekly", "JKUAT Lab / Hybrid", "PR merges, model retraining metrics, API test runs", "Brian, Role 2, Role 3, Role 4"],
        ["Bi-Weekly All-Hands", "Every 2 Weeks", "Physical at JKUAT", "Cross-team data-to-code alignment, patron progress demo", "All 6 Members"],
        ["Deployment Milestone", "Milestone (Week 4)", "JKUAT Desktop Lab", "On-premise hardware installation, WhatsApp live test", "All 6 Members (Mandatory)"],
    ]
    body.append(table(cadence_table))

    # Section 4: GitHub Protocol
    body.append(p(""))
    body.append(p("4. GitHub Collaboration & Branching Strategy", style="Heading1"))
    body.append(p("The project uses a structured Git workflow to prevent merge conflicts and maintain production stability:"))
    body.append(bullet("main: Production-ready code only. Direct commits are strictly disabled. Merges require PR approval from Brian.", bold_prefix="Protected Branch:"))
    body.append(bullet("develop: Integration staging branch where completed features are merged and tested.", bold_prefix="Staging Branch:"))
    body.append(bullet("tech/whatsapp-api: Owned by Role 2 for webhook and API handlers.", bold_prefix="Feature Branch:"))
    body.append(bullet("tech/fastapi-platform: Owned by Role 3 for server endpoints and Docker configs.", bold_prefix="Feature Branch:"))
    body.append(bullet("data/sheet-sanitization: Owned by Role 4 for master CSV cleanups.", bold_prefix="Feature Branch:"))
    body.append(bullet("data/literature-sourcing: Owned by Role 5 for PDF text additions to knowledge_base/.", bold_prefix="Feature Branch:"))
    body.append(bullet("model/catboost-tuning: Owned by Role 6 and Brian for training expansions.", bold_prefix="Feature Branch:"))

    body.append(p("Commit Message Conventions:", style="Heading3"))
    body.append(bullet("feat: New technical capability (e.g., 'feat: add WhatsApp webhook handler')"))
    body.append(bullet("data: Dataset additions or cleaning (e.g., 'data: normalize root length units to cm')"))
    body.append(bullet("docs: Research summaries or docs (e.g., 'docs: add ICRAF Vetiver technical bulletin')"))
    body.append(bullet("fix: Bug fixes (e.g., 'fix: handle missing salinity category in CatBoost inference')"))
    body.append(bullet("perf: Speed or hardware optimization (e.g., 'perf: configure 6 CPU threads for Ollama')"))

    # Section 5: Deployment Roadmap
    body.append(p(""))
    body.append(p("5. On-Premises JKUAT Deployment & WhatsApp Launch Roadmap", style="Heading1"))
    body.append(p("The project culminates in a complete on-premises deployment on high-performance desktop workstations at JKUAT:"))
    body.append(bullet("Phase 1 (Week 1 - 2): Master Google Sheet audit, PDF literature sourcing, and initial FastAPI endpoint creation.", bold_prefix="Milestone 1:"))
    body.append(bullet("Phase 2 (Week 2 - 3): WhatsApp Cloud API sandbox configuration, query router integration, and Golden Benchmark evaluation.", bold_prefix="Milestone 2:"))
    body.append(bullet("Phase 3 (Week 4): Physical assembly of all 6 team members at the JKUAT Computer Lab for on-premise installation.", bold_prefix="Milestone 3:"))

    body.append(p("JKUAT Workstation Setup Checklist (All Members Present):", style="Heading2"))
    body.append(bullet("Verify workstation CPU/GPU specs and install Ollama as a persistent background service."))
    body.append(bullet("Pull production models: qwen2.5:3b (or qwen2.5:1.5b) and nomic-embed-text."))
    body.append(bullet("Deploy the container stack using 'docker compose up -d' with persistent volume mounts."))
    body.append(bullet("Launch Cloudflare Tunnel to expose the FastAPI WhatsApp webhook over a secure public HTTPS endpoint."))
    body.append(bullet("Execute end-to-end live testing with student groups, faculty reviewers, and the project patron."))

    body.append(p(""))
    body.append(callout("Sign-Off Notice", "This charter establishes the operating guidelines for the VETIVER AI Platform development lifecycle. Any amendments to task allocations or schema definitions must be approved by the Project Lead (Brian)."))

    # Final XML document wrapper
    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    # Write to zip (.docx)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/document.xml", doc_xml)

    print(f"Successfully generated Word document: {filename} ({os.path.getsize(filename):,} bytes)")


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "VETIVER_Team_Charter_and_Operational_Plan.docx")
    create_docx(out_path)
