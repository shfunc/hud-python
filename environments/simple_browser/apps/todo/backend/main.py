from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import json

app = FastAPI(title="Sample API", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Item(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    completed: bool = False
    created_at: Optional[datetime] = None

class ItemCreate(BaseModel):
    title: str
    description: str
    completed: bool = False

# Database setup
def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            completed BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Routes
@app.get("/api/status")
def status():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/items", response_model=List[Item])
def get_items():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM items ORDER BY created_at DESC')
    items = [dict(row) for row in c.fetchall()]
    conn.close()
    return items

@app.post("/api/items", response_model=Item)
def create_item(item: ItemCreate):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO items (title, description, completed) VALUES (?, ?, ?)',
        (item.title, item.description, item.completed)
    )
    item_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return get_item(item_id)

@app.get("/api/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM items WHERE id = ?', (item_id,))
    item = c.fetchone()
    conn.close()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return dict(item)

@app.put("/api/items/{item_id}", response_model=Item)
def update_item(item_id: int, item: ItemCreate):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute(
        'UPDATE items SET title = ?, description = ?, completed = ? WHERE id = ?',
        (item.title, item.description, item.completed, item_id)
    )
    conn.commit()
    conn.close()
    
    if c.rowcount == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return get_item(item_id)

@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('DELETE FROM items WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    
    if c.rowcount == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"message": "Item deleted successfully"} 