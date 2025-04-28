from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
DATABASE_URL = "sqlite:///./expenses.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Expense Table
class Expense(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    amount = Column(Float)
    category = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

# Pydantic schemas
class ExpenseCreate(BaseModel):
    title: str
    amount: float
    category: str

class ExpenseResponse(ExpenseCreate):
    id: int
    date: datetime

    class Config:
        orm_mode = True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes

@app.post("/expenses/", response_model=ExpenseResponse)
def create_expense(expense: ExpenseCreate):
    db = SessionLocal()
    db_expense = Expense(**expense.dict())
    db.add(db_expense)
    db.commit()
    db.refresh(db_expense)
    db.close()
    return db_expense

@app.get("/expenses/", response_model=List[ExpenseResponse])
def read_expenses():
    db = SessionLocal()
    expenses = db.query(Expense).all()
    db.close()
    return expenses

@app.get("/expenses/{expense_id}", response_model=ExpenseResponse)
def read_expense(expense_id: int):
    db = SessionLocal()
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    db.close()
    if expense is None:
        raise HTTPException(status_code=404, detail="Expense not found")
    return expense

@app.put("/expenses/{expense_id}", response_model=ExpenseResponse)
def update_expense(expense_id: int, updated_expense: ExpenseCreate):
    db = SessionLocal()
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if expense is None:
        db.close()
        raise HTTPException(status_code=404, detail="Expense not found")

    for key, value in updated_expense.dict().items():
        setattr(expense, key, value)

    db.commit()
    db.refresh(expense)
    db.close()
    return expense

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: int):
    db = SessionLocal()
    expense = db.query(Expense).filter(Expense.id == expense_id).first()
    if expense is None:
        db.close()
        raise HTTPException(status_code=404, detail="Expense not found")
    
    db.delete(expense)
    db.commit()
    db.close()
    return {"message": "Expense deleted successfully"}