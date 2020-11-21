# Fake_News_Detection
A respository consisting of a fake news detection methods tried over the binary classes of NELA-GT dataset.

### Demonstration
The proposed solution to the issue concerned with fake news includes the use of a web app framework Streamlit, which presents the user with an interactive dashboard where they can see the news classification in action. The user can enter the news title/content, which gets redirected to our best performing ML model in the back end and classifies the said news title/content into fake or real. </br>

(streamlit_main.ipynb has all these steps; can dierctly execute that)</br>
To run or deploy the Streamlit app from Google’s Colaboratory, first install it using:</br>
                !pip install streamlit

Then we'll have to use Ngrok for providing secure tunnels from the local system to the public. Follow these commands:</br>
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip </br>
!unzip ngrok-stable-linux-amd64.zip </br>
get_ipython().system_raw('./ngrok http 8501 &') </br>
!curl -s http://localhost:4040/api/tunnels | python3 -c \ </br>
'import sys, json; print("Access the app from the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])' <br><br>

Then simply execute the respective python file "streamlit.py" to deploy on the Streamlit app </br>
!streamlit run /content/streamlit.py

### Youtube link --> https://www.youtube.com/watch?v=2c8lJSKtQHc&t=54s
The above link is for a video on complete description of our work.
