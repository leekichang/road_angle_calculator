f = open('./api_key.txt', 'r', encoding='utf-8')
API_KEY = f.read().strip()
f.close()