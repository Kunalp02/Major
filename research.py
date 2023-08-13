import requests
paper_url = "https://ieeexplore.ieee.org/document/9524580"
response = requests.get(paper_url)
if response.status_code == 200:
    with open("paper.pdf", "wb") as pdf_file:
        pdf_file.write(response.content)
        print("Paper downloaded successfully.")
else:
    print("Paper could not be downloaded. Status code:", response.status_code)


with open("paper.pdf", "wb") as pdf_file:
    pdf_file.write(response.content)
