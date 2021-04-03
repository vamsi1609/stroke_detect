import streamlit as st
from streamlit.hashing import _CodeHasher
import pickle
import numpy as np
from PIL import Image

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


def main():
    state = _get_state()
    pages = {
        
        "Home": page_settings,
        "Result": page_dashboard,
    }

    st.sidebar.title("Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))
    
    # Display the selected page with the session state
    pages[page](state)
         
    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def page_dashboard(state):
    st.title("Result page")
    display_state_values(state)


def page_settings(state):
    st.title(
        'Stroke Detection via Machine Learning'
    )
    st.write("---")
    state.name = st.text_input('Input your name')
    gender = st.radio("Gender",('Female', 'Male'))
    state.gender = 0 if gender =='Female' else 1

    state.age = st.slider("Age",1,100,20)
    hypertension = st.radio("Hypertension",('Yes', 'No'))
    state.hypertension = 0 if hypertension =='No' else 1

    heart_disease = st.radio("Heart Disease",('Yes', 'No'))
    state.heart_disease = 0 if heart_disease =='No' else 1

    ever_married = st.radio("Marital Status",('Yes', 'No'))
    state.ever_married = 0 if ever_married =='No' else 1

    work_type = st.selectbox("Work Type", ("Govt_job","Never_worked","Private","Self_employed","children"))
    if(work_type=="Govt_job"):
        state.work_type=0
    elif(work_type=="Never_worked"):
        state.work_type=1
    elif(work_type=="Private"):
        state.work_type=2
    elif(work_type=="Private"):
        state.work_type=3
    else:
        state.work_type=4


    residence_type = st.radio("Residance type",('Rural', 'Urban'))
    state.residence_type = 0 if residence_type =='Rural' else 1

    state.avg_glucose_level = st.slider("Average Glucose Level", 40.00, 290.00)
    state.bmi = st.slider("BMI",5.0,110.0)
    smoking_status = st.selectbox("Smoking Status",('Unknown', 'formerly smoked', 'never smoked', 'smokes'))

    if(smoking_status=="Unknown"):
        state.smoking_status=0
    elif(smoking_status=="formerly smoked"):
        state.smoking_status=1
    elif(smoking_status=="never smoked"):
        state.smoking_status=2
    else:
        state.smoking_status=3

    x_pred = np.array([[state.gender,state.age,state.hypertension,state.heart_disease,state.ever_married,state.work_type,state.residence_type,state.avg_glucose_level,state.bmi,state.smoking_status]])
    Filename = 'models/RF_KNN_Model.pkl'
    with open(Filename, 'rb') as file:  
        KNN_Model = pickle.load(file)

    y = KNN_Model.predict(x_pred)

    state.y = "No" if y==0 else "Yes"
    st.write(" __________________________ ")
    if st.button("Predict"):
            if (state.y=="No"):
                st.write(f"""
                    Proceed to Result section :) 
                """)
            else:
                st.write(f"""
                    Proceed to Result section :( 
                """)
    st.write("Note: Proceeding to Results will clear all fields")
def display_state_values(state):
    if (state.y=="No"):
        st.write(f"""
            ## Hi {state.name}, you are at a lower risk of having a stroke.
        """)
        image = Image.open('happy.jpg')
        st.image(image)

    else:
        st.write(f"""
            ## Hi {state.name}, sorry you might get or might have had a stroke
        """)
        image = Image.open('worry.jpg')
        st.image(image)
    if st.button("Clear state"):
        state.clear()


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()