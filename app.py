import os
import requests
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# We use Mistral 7B because T5 seems to be having routing issues on the free tier today.
# Mistral is very smart and usually always active.
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
)

HF_API_KEY = os.environ.get("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def query_huggingface(payload):
    print(f"⚡ Sending to: {API_URL}")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # DEBUG: Print what we got back
        print(f"📥 Status: {response.status_code}")
        print(f"📥 Body: {response.text}")

        # Handle "Model Loading" (Common on free tier)
        if response.status_code == 503:
            return {
                "error": "Model is loading",
                "estimated_time": response.json().get("estimated_time", 20),
            }

        return response.json()
    except Exception as e:
        print(f"Network Error: {e}")
        return {"error": str(e)}


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"🎤 Prompt: {voice_prompt}")

        # Mistral uses [INST] format
        # We give it a strict persona to only output SQL
        prompt = (
            f"<s>[INST] You are a SQL expert. Convert this question to SQL. "
            f"Return ONLY the SQL query. Do not explain.\n\n"
            f"Question: {voice_prompt} [/INST]\n"
            f"SELECT"
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "return_full_text": False,
                "temperature": 0.1,
            },
            "options": {"wait_for_model": True},
        }

        output = query_huggingface(payload)

        # Logic to extract text safely
        generated_sql = "Error"

        if isinstance(output, list) and len(output) > 0:
            # Mistral returns list of dicts
            raw_text = output[0].get("generated_text", "")
            # We added "SELECT" to the prompt, so we prepend it back
            generated_sql = "SELECT " + raw_text.strip()

        elif isinstance(output, dict) and "error" in output:
            error_msg = output.get("error")
            print(f"⚠️ API Error: {error_msg}")

            if "loading" in str(error_msg).lower():
                return jsonify(
                    {
                        "status": "error",
                        "message": "AI is warming up... Try again in 20s",
                    }
                ), 503

            return jsonify(
                {"status": "error", "message": f"AI Error: {error_msg}"}
            ), 500

        print(f"🤖 SQL: {generated_sql}")
        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
