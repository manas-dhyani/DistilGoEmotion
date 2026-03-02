import google.generativeai as genai

genai.configure(api_key="AIzaSyBCb1SUyeNoDMPWOpoSitdoaGOptqGvy-Q")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Write a poem about emotions.")

print(response.text)
