"""This module contains the general questions chain."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from llms.utils import get_o3_mini_east_2_llm


class GeneralFacts(BaseModel):
    """Model for the general facts section of the medical analysis."""

    name: Optional[str] = Field(description="Patient's full name")
    age: Optional[str] = Field(
        description="Patient's age, just the number, use the latest age"
    )
    gender: Optional[str] = Field(description="Patient's gender")
    height: Optional[str] = Field(description="Patient's height (in feet'inches\")")
    weight: Optional[str] = Field(description="Patient's weight (in pounds)")
    occupation: Optional[str] = Field(description="Patient's occupation")
    residence: Optional[str] = Field(description="Patient' state residence")
    smoking_status: Optional[str] = Field(
        description="Patient's smoking status details"
    )


class FinalAnalysis(BaseModel):
    general: GeneralFacts = Field(description="General facts about the patient")
    overview: str = Field(
        description="Free-form text providing an overview of the patient's health"
    )
    labs_and_measurements: str = Field()
    substance_usage: str = Field()
    medical_conditions: str = Field()
    prescription_usage: str = Field()
    family_history: str = Field()
    genetics: str = Field()
    favorable_factors: str = Field()
    diagnostics: str = Field()


llm = get_o3_mini_east_2_llm()


system = """Formatting re-enabled
You are tasked with analyzing a comprehensive medical document that contains various types of reports and data related to a single individual's health. The document contains different medical test results, patient history, and other vital information.

You will receive an array of documents, where each item represents the information extracted from a page of the medical document. You can consider the indexes of the array as the page numbers.
Your goal is to extract, organize, and present critical information based on the detailed categories below into a cohesive, structured report.

### Rules:
- Always get the most recent values using the date of the report
- Use bullets (e.g. -) to show multiple data points under each category, never add titles.
- Ensure that all sections are filled based on the available information extracted from the array. If certain details are not present in the document, **omit that specific bullet point** and focus only on the available information.
- The final output should be a single, structured report.
- Do not omit empty sections, if a section is empty, state that no information is available.

### Final Output Structure:
Return the structured report as follows:
1. **General Facts:**
   - **Name**:
   - **Age**: Only the number, most recent age if possible
   - **Gender**:
   - **Height**: (in feet'inches") convert if necessary, most recent height if possible
   - **Weight**: (in pounds) convert if necessary, most recent weight if possible
   - **Occupation**:
   - **Residence**:
   - **Smoking Status**:

2. **Labs and Measurements:**
   - **HbA1c**: Last reading, and any out-of-range values.
   - **LDL**: Last reading, and any out-of-range values.
   - **HDL**: Last reading, and any out-of-range values.
   - **Total Cholesterol**: Last reading, and any out-of-range values.
   - **Triglycerides**: Last reading, and any out-of-range values.
   - **Blood Pressure (Systolic and Diastolic)**: Last reading, and any out-of-range values.
   - **Antigen Test**: Last reading, and any out-of-range values.

3. **Substance Usage:**
   - Tobacco use, Alcohol use, Marijuana use, any other substance usage.

4. **Medical Conditions:**
   - **Most Common Conditions**: Diagnoses of common conditions (Diabetes, Sleep Apnea, Hypertension, Depression, COPD, Arthritis, Asthma, chronic injuries) and management.
   - **Cancer History**: Type of cancer, diagnosis date, and treatment status.
   - **Surgery History**: Surgical interventions, including dates and reasons.
   - **Other Medical Conditions**: Any additional conditions cited.

5. **Prescription Usage:**
   - **Current Prescriptions**: List of current medications, including dosage and conditions treated.
   - **Previous Medications**: Summary of historical medications and reasons for discontinuation.

6. **Family History:**
   - **Parents' Ages**: Current age of parents and siblings. Age at death and cause for any deceased family members.

7. **Genetics:**
   - **Genetic Factors**: Any positive or negative genetic factors included in the records.

8. **Other Considerations:**
   - **Diagnostics**: Key takeaways from any other diagnostic tests included.
   - **Favorable Factors**: Any healthy behaviors or favorable conditions mentioned in their medical records.

output: make it readable, markdown is preferred
"""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            """documents: {documents}""",
        ),
    ]
)

medical_analyzer_chain = prompt | llm.with_structured_output(
    FinalAnalysis,
    method="function_calling",
    strict=True,
)
