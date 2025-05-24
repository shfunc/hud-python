from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

load_dotenv()

# Import hud for telemetry
import hud
from hud.telemetry.context import buffer_mcp_call, flush_buffer

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    # Create simple configuration
    config = {
    "mcpServers": {
        "airbnb": {
        "command": "npx",
        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        "env": {
            "IGNORE_ROBOTS_TXT": "true"
        }
        }
    }
    }
    
    try:
        print("Creating MCPClient...")
        client = MCPClient.from_dict(config)
        print("MCPClient created")
        
        print("Creating LLM...")
        llm = ChatOpenAI(model="gpt-4o")
        print("LLM created")
        
        print("Creating MCPAgent...")
        agent = MCPAgent(llm=llm, client=client, max_steps=5, verbose=False)
        print("MCPAgent created")
        
        # Define trace ID
        task_run_id = "test-run-1"
        print(f"Starting trace with ID: {task_run_id}")
        
        # Run the query inside a trace context
        async with hud.trace(attributes={"query": "Find the best restaurant in San Francisco"}):
            print("Running agent with query...")
            result = await agent.run(
                "Find me a nice place to stay in Barcelona for 2 adults "
                "for a week in August. I prefer places with a pool and "
                "good reviews. Show me the top 3 options.",
                max_steps=5,
            )
            print(f"Result: {result}")
            
            # Access the telemetry buffer (for debugging purposes)
            telemetry_data = flush_buffer()
            print(f"\nCollected {len(telemetry_data)} telemetry records")

            telemetry_data = [record.model_dump() for record in telemetry_data]
            
            # Print all records to see what was captured
            seen_types = set()
            print("\n=== DETAILED TELEMETRY RECORDS ===")
            for i, record in enumerate(telemetry_data[:10]):  # Just show first 10 to avoid overwhelming output
                call_type = record.get("call_type", "unknown")
                direction = record.get("direction", "unknown")
                method = record.get("method", "unknown")
                status = record.get("status", "unknown")
                is_response = record.get("is_response_or_error", False)
                
                print(f"\nRecord #{i+1}:")
                print(f"  Type: {call_type}")
                print(f"  Direction: {direction}")
                print(f"  Method: {method}")
                print(f"  Status: {status}")
                if is_response:
                    print(f"  Is Response: {is_response}")
                    
                # Print timestamps if available
                if "timestamp" in record:
                    print(f"  Timestamp: {record['timestamp']}")
                elif "start_time" in record:
                    print(f"  Start Time: {record['start_time']}")
                    if "end_time" in record:
                        print(f"  End Time: {record['end_time']}")
                        print(f"  Duration: {record.get('duration', 'N/A')} sec")
                
                # Add any other interesting details
                for key in ["error", "error_type", "is_error", "stream_event"]:
                    if key in record:
                        print(f"  {key}: {record[key]}")
            
            if len(telemetry_data) > 10:
                print(f"\n... {len(telemetry_data) - 10} more records...")
            
            print("\n=== RECORD TYPE SUMMARY ===")
            
            # Print summary of record types
            seen_types = set()
            for record in telemetry_data:
                call_type = record.get("call_type", "unknown")
                direction = record.get("direction", "unknown")
                # Only print first occurrence of each call type to avoid overwhelming output
                if (call_type, direction) not in seen_types:
                    seen_types.add((call_type, direction))
                    print(f"\nRecord Type: {call_type}")
                    print(f"  Status: {record.get('status', 'unknown')}")
                    print(f"  Direction: {direction}")
                    # Print any other interesting fields
                    if "method" in record:
                        print(f"  Method: {record['method']}")
                    if "is_response_or_error" in record:
                        print(f"  Is Response/Error: {record['is_response_or_error']}")
            
            # Check if our manual test call is in the buffer
            print("\nChecking for manual test call...")
            manual_test_calls = [r for r in telemetry_data if r.get("call_type") == "manual.test"]
            print(f"Found {len(manual_test_calls)} manual test calls")
            if manual_test_calls:
                print(f"Manual test call attributes: {manual_test_calls[0]}")
                
            # Count unique call types to see what operations were captured
            call_types = {}
            for record in telemetry_data:
                call_type = record.get("call_type", "unknown")
                call_types[call_type] = call_types.get(call_type, 0) + 1
            
            print("\nCall type distribution:")
            for call_type, count in call_types.items():
                print(f"  {call_type}: {count}")
                
            # Instead of flushing, we'll put the telemetry back into the buffer for export
            # by buffering them again
            for record in telemetry_data:
                buffer_mcp_call(record)
        
        # Clean up
        print("Closing connections...")
        try:
            if hasattr(client, "close_all_sessions"):
                await client.close_all_sessions()
            elif hasattr(agent, "close"):
                await agent.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
