# Step 1 :

### Installing python requirements

#### Create conda environment 
`conda create -n "env_name python==3.8"  - Compatible in python 3.8`

#### Activate Environment
`conda activate "env_name"`

#### Install dependencies
`pip install -r requirements.txt`

# Step 2 :

### Run streamlit server
`streamlit run chatbot.py`

#Step 3 :

### Create Vector Database in Pinecone and Create new api key then paste it in chatbot.py

#Step 4 :

### Upload the Book in pdf format

<img src="https://github.com/danielprinceD/OpenBook-AI/blob/main/How%20to/1.Upload%20PDF.png">

#Step 5 :

### Embedding Process takes place depends upon the corpus (text segments ) in the book

<ing src="https://github.com/danielprinceD/OpenBook-AI/blob/main/How%20to/2.Embedding%20Process.png">

#Step 6 :

### Storing Vector in Database (NOTE: Free trail in pinecone only stores 100K Vectors only)

<ing src="https://github.com/danielprinceD/OpenBook-AI/blob/main/How%20to/3.Storing%20Vector%20in%20Database.png">

#Step 7 :

### Model is Ready You can start Chat with your book

<ing src="https://github.com/danielprinceD/OpenBook-AI/blob/main/How%20to/4.Model%20Ready.png">
