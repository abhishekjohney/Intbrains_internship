from pydantic import BaseModel
from typing import List, Optional

class Project(BaseModel):
    Name: str
    Description: Optional[str] = None
    Technologies_Used: List[str] = []

class WorkExperience(BaseModel):
    Role: str
    Company: Optional[str] = None
    Duration_Months: Optional[int] = None
    Description: Optional[str] = None

class EducationEntry(BaseModel):
    Title: str
    Institution: Optional[str] = None
    Duration: Optional[str] = None

class ResumeData(BaseModel):
    Name: Optional[str] = None
    Email: Optional[str] = None
    Primary_Role: Optional[str] = None
    Programming_Languages: List[str] = []
    Frameworks_Tools: List[str] = []
    Years_of_Experience: Optional[int] = None
    Projects: List[Project] = []
    Total_Projects_Count: int = 0
    Experiences: List[WorkExperience] = []
    Education: List[EducationEntry] = []
