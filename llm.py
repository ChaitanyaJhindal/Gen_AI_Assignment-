import os

from openai import OpenAI


MODELS = [
	"openai/gpt-oss-120b",
	"llama-3.3-70b-versatile",
	"llama-3.1-8b-instant",
]


def load_dotenv_if_available() -> None:
	try:
		from dotenv import load_dotenv

		load_dotenv()
	except Exception:
		env_path = ".env"
		if not os.path.exists(env_path):
			return
		with open(env_path, "r", encoding="utf-8") as env_file:
			for line in env_file:
				raw = line.strip()
				if not raw or raw.startswith("#") or "=" not in raw:
					continue
				key, value = raw.split("=", 1)
				key = key.strip()
				value = value.strip().strip('"').strip("'")
				if key and key not in os.environ:
					os.environ[key] = value


def build_client() -> OpenAI:
	api_key = os.environ.get("GROQ_API_KEY")
	if not api_key:
		raise ValueError("Missing GROQ_API_KEY in environment or .env")
	return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def ask_with_models(client: OpenAI, query: str) -> dict[str, str]:
	results: dict[str, str] = {}
	for model in MODELS:
		try:
			response = client.responses.create(input=query, model=model)
			results[model] = response.output_text
		except Exception as exc:
			results[model] = f"Error: {exc}"
	return results


if __name__ == "__main__":
	load_dotenv_if_available()
	client = build_client()

	print("Multi-LLM Query Runner")
	print("Type your question. Type 'exit' to quit.\n")

	while True:
		user_query = input("Enter query: ").strip()
		if not user_query:
			continue
		if user_query.lower() in {"exit", "quit"}:
			print("Exiting.")
			break

		answers = ask_with_models(client, user_query)
		print("\n=== Model Responses ===")
		for model_name, answer in answers.items():
			print(f"\n[{model_name}]\n{answer}\n")
