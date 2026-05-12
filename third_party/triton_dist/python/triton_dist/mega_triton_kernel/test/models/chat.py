################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import socket
import json
import argparse
import time


class ChatClient:

    def __init__(self, server_host: str = "localhost", server_port: int = 9999, timeout: int = 300):
        """Initialize the chat client"""
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.connected = False

    def connect(self):
        """Establish a connection to the server"""
        try:
            if self.connected:
                return True

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"Connected to server {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    pass
                self.socket = None
            return False

    def disconnect(self):
        """Disconnect from the server"""
        if self.connected and self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            self.connected = False
            print("Disconnected from server")

    def send_prompt(self, prompt: str) -> tuple:
        """Send a prompt to the server and get the response with timing information"""
        if not self.connected:
            if not self.connect():
                return "Sorry, failed to connect to the server.", 0, 0, 0

        try:
            # Record start time
            start_time = time.time()

            # Construct the request data
            data = json.dumps({"prompt": prompt}).encode('utf-8')

            # Send the data
            self.socket.sendall(data)

            # Receive the response - loop until the entire response is received
            response_chunks = []
            while True:
                try:
                    chunk = self.socket.recv(4096)
                    if not chunk:
                        # Connection closed by server
                        self.connected = False
                        print("Server closed the connection")
                        break
                    response_chunks.append(chunk)

                    # Try to parse JSON to check if the full response is received
                    try:
                        response_data = b''.join(response_chunks).decode('utf-8')
                        json.loads(response_data)
                        # If parsing succeeds, the full response is received
                        break
                    except json.JSONDecodeError:
                        # Continue receiving data
                        continue
                except socket.timeout:
                    print("Socket timeout while receiving response")
                    self.connected = False
                    break

            # Record end time
            end_time = time.time()
            total_time = end_time - start_time

            if not response_chunks:
                return "Sorry, an error occurred while receiving the response from the server.", 0, total_time, 0

            response_data = b''.join(response_chunks).decode('utf-8')

            # Parse the JSON response
            try:
                response = json.loads(response_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding server response: {e}")
                print(f"Response data: {response_data}")
                self.connected = False
                return f"Sorry, an error occurred while parsing the server response: {e}", 0, total_time, 0

            # Calculate tokens per second
            processing_time = response.get("processing_time", total_time)
            response_text = response.get("response", "")

            # Approximate token count (1 token ~= 4 characters)
            token_count = max(1, len(response_text) // 4)
            tokens_per_second = token_count / processing_time if processing_time > 0 else 0

            return response_text, token_count, processing_time, tokens_per_second

        except Exception as e:
            print(f"Communication error: {e}")
            self.connected = False
            return f"Sorry, an error occurred while communicating with the server: {e}", 0, 0, 0

    def start_chat(self):
        """Start the interactive chat interface with timing information"""
        print("==== Chat Assistant ====")
        print("Type 'exit' to end the chat")
        print("----------------------------")

        try:
            # Connect to the server
            if not self.connect():
                print("Failed to connect to the server. Chat session terminated.")
                return

            while True:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check exit condition
                if user_input.lower() == "exit":
                    print("Thank you for using. Goodbye!")
                    break

                if not user_input:
                    continue

                # Show processing status
                print("AI is thinking...", end="", flush=True)

                # Send request and get response with timing
                response, tokens, time_taken, tokens_per_sec = self.send_prompt(user_input)

                # Clear processing status
                print("\r" + " " * 20 + "\r", end="")

                # Get current time in HH:MM:SS format
                current_time = time.strftime("%H:%M:%S")

                # Display response with time and speed
                print(f"AI ({current_time}): {response}")
                print(f"Tokens: {tokens} | Time: {time_taken:.2f}s | Speed: {tokens_per_sec:.2f} tokens/second")

        finally:
            # Disconnect from the server
            self.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str, help="Server host")
    parser.add_argument("--port", default=9999, type=int, help="Server port")
    args = parser.parse_args()

    client = ChatClient(args.host, args.port)
    client.start_chat()


if __name__ == "__main__":
    main()
