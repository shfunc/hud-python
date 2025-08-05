'use client'

import { useState, useEffect } from 'react'

interface Item {
  id: number
  title: string
  description: string
  completed: boolean
  created_at: string
}

export default function Home() {
  const [items, setItems] = useState<Item[]>([])
  const [newTitle, setNewTitle] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchItems()
  }, [])

  const fetchItems = async () => {
    try {
      const response = await fetch('/api/items')
      const data = await response.json()
      setItems(data)
    } catch (error) {
      console.error('Error fetching items:', error)
    } finally {
      setLoading(false)
    }
  }

  const createItem = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newTitle.trim()) return

    try {
      const response = await fetch('/api/items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newTitle,
          description: newDescription,
          completed: false
        })
      })
      
      if (response.ok) {
        setNewTitle('')
        setNewDescription('')
        fetchItems()
      }
    } catch (error) {
      console.error('Error creating item:', error)
    }
  }

  const toggleItem = async (id: number, item: Item) => {
    try {
      const response = await fetch(`/api/items/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...item,
          completed: !item.completed
        })
      })
      
      if (response.ok) {
        fetchItems()
      }
    } catch (error) {
      console.error('Error updating item:', error)
    }
  }

  const deleteItem = async (id: number) => {
    try {
      const response = await fetch(`/api/items/${id}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        fetchItems()
      }
    } catch (error) {
      console.error('Error deleting item:', error)
    }
  }

  return (
    <main className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">Sample Todo App</h1>
        
        {/* Create Form */}
        <form onSubmit={createItem} className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Add New Item</h2>
          <div className="space-y-4">
            <input
              type="text"
              placeholder="Title"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <textarea
              placeholder="Description"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
            />
            <button
              type="submit"
              className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 transition-colors"
            >
              Add Item
            </button>
          </div>
        </form>

        {/* Items List */}
        <div className="space-y-4">
          {loading ? (
            <p className="text-gray-500">Loading...</p>
          ) : items.length === 0 ? (
            <p className="text-gray-500">No items yet. Create one above!</p>
          ) : (
            items.map((item) => (
              <div
                key={item.id}
                className="bg-white rounded-lg shadow-md p-6 flex items-start justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={item.completed}
                      onChange={() => toggleItem(item.id, item)}
                      className="mr-3 h-5 w-5 text-blue-600"
                    />
                    <h3 className={`text-lg font-semibold ${item.completed ? 'line-through text-gray-500' : 'text-gray-900'}`}>
                      {item.title}
                    </h3>
                  </div>
                  {item.description && (
                    <p className="mt-2 text-gray-600 ml-8">{item.description}</p>
                  )}
                  <p className="mt-2 text-sm text-gray-400 ml-8">
                    Created: {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
                <button
                  onClick={() => deleteItem(item.id)}
                  className="ml-4 text-red-500 hover:text-red-700"
                >
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </main>
  )
} 