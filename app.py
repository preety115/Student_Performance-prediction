
import os
import time
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf

# --------------------------------
# 1. Load trained model and files
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "student_performance_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "student_performance_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "student_performance_features.pkl")

required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
missing_files = [p for p in required_files if not os.path.exists(p)]

if missing_files:
    raise FileNotFoundError(
        "Required files are missing:\n" + "\n".join(missing_files)
    )

print("Loading trained model and preprocessing files...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
feature_columns = list(joblib.load(FEATURES_PATH))

# Warm up the model once during startup
dummy_input = np.zeros(
    (1, len(feature_columns)), dtype=np.float32
)
model.predict(dummy_input, verbose=0)

print("Model loaded and initialized successfully.")


# --------------------------------
# 2. Student prediction function
# --------------------------------
def predict_student(
    school, sex, age, address, famsize, Pstatus,
    Medu, Fedu, Mjob, Fjob, reason, guardian,
    traveltime, studytime, failures, schoolsup,
    famsup, paid, activities, nursery, higher,
    internet, romantic, famrel, freetime, goout,
    Dalc, Walc, health, absences
):
    start_time = time.perf_counter()

    try:
        student = pd.DataFrame([{
            "school": school,
            "sex": sex,
            "age": int(age),
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": int(Medu),
            "Fedu": int(Fedu),
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": int(traveltime),
            "studytime": int(studytime),
            "failures": int(failures),
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": int(famrel),
            "freetime": int(freetime),
            "goout": int(goout),
            "Dalc": int(Dalc),
            "Walc": int(Walc),
            "health": int(health),
            "absences": int(absences)
        }])

        # Encode categorical features as in training
        encoded = pd.get_dummies(
            student, drop_first=True, dtype=int
        )

        # Match the original training feature names and order
        encoded = encoded.reindex(
            columns=feature_columns, fill_value=0
        )

        # Apply the saved scaler
        scaled = scaler.transform(encoded)
        scaled = np.asarray(scaled, dtype=np.float32)

        # Predict the final grade
        prediction_array = model.predict(
            scaled, verbose=0
        )
        prediction = float(
            np.asarray(prediction_array).reshape(-1)[0]
        )

        # Grade range: 0 to 20
        prediction = float(np.clip(prediction, 0, 20))

        elapsed = time.perf_counter() - start_time
        print(f"Prediction completed in {elapsed:.3f} seconds.")

        return (
            f"Estimated Final Grade (G3): {prediction:.2f} / 20\n\n"
            "This is an estimate, not a guaranteed result."
        )

    except Exception as error:
        print(f"Prediction error: {error}")
        return f"Prediction failed: {error}"


# --------------------------------
# 3. Build the Gradio interface
# --------------------------------
with gr.Blocks(title="Student Performance Prediction") as demo:

    gr.Markdown(
        """
        # Student Performance Prediction Using ANN

        Enter student information to estimate the final grade (G3).
        The model predicts grades on a scale from 0 to 20.
        """
    )

    with gr.Row():

        # Student Information
        with gr.Column():
            gr.Markdown("### Student Information")

            school = gr.Dropdown(
                ["GP", "MS"],
                value="GP",
                label="School"
            )

            # Interface label is Gender.
            # Internal dataset feature remains "sex".
            sex = gr.Dropdown(
                ["F", "M"],
                value="F",
                label="Gender"
            )

            age = gr.Slider(
                15, 22, value=17, step=1,
                label="Age"
            )

            address = gr.Dropdown(
                ["U", "R"], value="U",
                label="Address: Urban/Rural"
            )

            famsize = gr.Dropdown(
                ["GT3", "LE3"], value="GT3",
                label="Family Size"
            )

            Pstatus = gr.Dropdown(
                ["T", "A"], value="T",
                label="Parents' Cohabitation"
            )

            Medu = gr.Slider(
                0, 4, value=3, step=1,
                label="Mother's Education (0–4)"
            )

            Fedu = gr.Slider(
                0, 4, value=2, step=1,
                label="Father's Education (0–4)"
            )

            Mjob = gr.Dropdown(
                ["teacher", "health", "services", "at_home", "other"],
                value="other",
                label="Mother's Job"
            )

            Fjob = gr.Dropdown(
                ["teacher", "health", "services", "at_home", "other"],
                value="other",
                label="Father's Job"
            )

            reason = gr.Dropdown(
                ["home", "reputation", "course", "other"],
                value="course",
                label="Reason for Choosing School"
            )

            guardian = gr.Dropdown(
                ["mother", "father", "other"],
                value="mother",
                label="Guardian"
            )

        # Study and Lifestyle
        with gr.Column():
            gr.Markdown("### Study and Lifestyle")

            traveltime = gr.Slider(
                1, 4, value=1, step=1,
                label="Travel Time (1–4)"
            )

            studytime = gr.Slider(
                1, 4, value=2, step=1,
                label="Weekly Study Time (1–4)"
            )

            failures = gr.Slider(
                0, 3, value=0, step=1,
                label="Past Class Failures"
            )

            schoolsup = gr.Dropdown(
                ["yes", "no"], value="no",
                label="Extra School Support"
            )

            famsup = gr.Dropdown(
                ["yes", "no"], value="yes",
                label="Family Educational Support"
            )

            paid = gr.Dropdown(
                ["yes", "no"], value="no",
                label="Extra Paid Classes"
            )

            activities = gr.Dropdown(
                ["yes", "no"], value="yes",
                label="Extracurricular Activities"
            )

            nursery = gr.Dropdown(
                ["yes", "no"], value="yes",
                label="Attended Nursery School"
            )

            higher = gr.Dropdown(
                ["yes", "no"], value="yes",
                label="Wants Higher Education"
            )

            internet = gr.Dropdown(
                ["yes", "no"], value="yes",
                label="Internet Access"
            )

            romantic = gr.Dropdown(
                ["yes", "no"], value="no",
                label="Romantic Relationship"
            )

        # Family, Health and Attendance
        with gr.Column():
            gr.Markdown("### Family, Health and Attendance")

            famrel = gr.Slider(
                1, 5, value=4, step=1,
                label="Family Relationship (1–5)"
            )

            freetime = gr.Slider(
                1, 5, value=3, step=1,
                label="Free Time (1–5)"
            )

            goout = gr.Slider(
                1, 5, value=3, step=1,
                label="Going Out (1–5)"
            )

            Dalc = gr.Slider(
                1, 5, value=1, step=1,
                label="Workday Alcohol Consumption (1–5)"
            )

            Walc = gr.Slider(
                1, 5, value=2, step=1,
                label="Weekend Alcohol Consumption (1–5)"
            )

            health = gr.Slider(
                1, 5, value=4, step=1,
                label="Health Status (1–5)"
            )

            absences = gr.Slider(
                0, 50, value=2, step=1,
                label="School Absences"
            )

    predict_button = gr.Button(
        "Predict Final Grade",
        variant="primary"
    )

    result = gr.Textbox(
        label="Prediction Result",
        lines=3
    )

    predict_button.click(
        fn=predict_student,
        inputs=[
            school, sex, age, address, famsize, Pstatus,
            Medu, Fedu, Mjob, Fjob, reason, guardian,
            traveltime, studytime, failures, schoolsup,
            famsup, paid, activities, nursery, higher,
            internet, romantic, famrel, freetime, goout,
            Dalc, Walc, health, absences
        ],
        outputs=result
    )


# --------------------------------
# 4. Launch local application
# --------------------------------
if __name__ == "__main__":
   

 demo.launch(
    server_name="127.0.0.1",
    server_port=7861,
    share=False
)
