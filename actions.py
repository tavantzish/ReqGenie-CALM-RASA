from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import openai 

class ActionValidateProblemDescription(Action):
    def name(self) -> Text:
        return "action_validate_problem_description"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]: 
        project_problem = tracker.get_slot("problem")

        if project_problem is None:
            dispatcher.utter_message(
                text="It seems you haven't provided a problem description yet. Can you please describe the issue?"
            )
            return []

        validation_prompt = f"Evaluate the following problem description: '{project_problem}'\n\Mention explicitly whether the problem description is clear, reasonable, and specific, or it is vague, unclear and has room for improvement? Provide any suggestions, if necessary."

        # Call GPT-4 (LLM) to validate the problem and goal
        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()
            dispatcher.utter_message(
                text=f"Thank you! This is a brief feedback on your problem description: {llm_feedback}"
            )
            return [
                SlotSet("proposed_problem_description", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your inputs. Please try again."
            )
            return []

class ActionAdjustProblemDescription(Action):
    def name(self) -> Text:
        return "action_adjust_problem_description"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]: 
        confirmation = tracker.get_slot("problem_confirmation")
        proposed_description = tracker.get_slot("proposed_problem_description")
        current_problem_description = tracker.get_slot("problem")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed problem description yet. Could you please provide your problem description of the project?"
            )
            return [
                SlotSet("problem", None)
            ]
        validation_prompt = f"Based on the user's confirmation and the proposed problem description, can you update the existing problem description of the system?\n\nConfirmation: '{confirmation}'\nProposed problem description: '{proposed_description}'\nExisting problem description: '{current_problem_description}'"

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final problem description of the project is: {llm_feedback}"
            )
            return [
                SlotSet("problem", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error adjusting the problem description. Please try again."
            )
            return []

class ActionValidateProjectGoal(Action):
    def name(self) -> Text:
        return "action_validate_project_goal"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        project_problem = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")

        if project_goal is None:
            dispatcher.utter_message(
                text="It seems you haven't provided a project goal yet. Can you please describe the objective of your project?"
            )
            return []

        validation_prompt = f"Evaluate the following project goal:\nGoal: {project_goal}\nBased on the problem description: {project_problem}\n Is the goal clear, reasonable, and specific? Does the goal align with the problem description? Suggest improvements for the goal description, if necessary."

        # Call GPT-4 (LLM) to validate the problem and goal
        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"Thank you! This is a brief feedback on your project goal description: {llm_feedback}"
            )
            return [
                SlotSet("proposed_project_goal", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your inputs. Please try again."
            )
            return []

class ActionAdjustProjectGoal(Action):
    def name(self) -> Text:
        return "action_adjust_project_goal"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]: 
        confirmation = tracker.get_slot("goal_confirmation")
        proposed_project_goal = tracker.get_slot("proposed_project_goal")
        current_project_goal = tracker.get_slot("goal")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed project goal yet. Could you please provide your project goal description of the project?"
            )
            return [
                SlotSet("goal", None)
            ]       
        validation_prompt = f"Based on the user's confirmation and the proposed project goal description, can you update the existing project goal of the system?\n\nConfirmation: '{confirmation}'\nProposed project goal: '{proposed_project_goal}'\nExisting project goal: '{current_project_goal}'"

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final project goal description is: {llm_feedback}"
            )
            return [
                SlotSet("goal", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error adjusting the project goal description. Please try again."
            )
            return []


class ActionValidatePrimaryStakeholders(Action):
    def name(self) -> Text:
        return "action_validate_primary_stakeholders"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        project_problem = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")
        primary_stakeholders = tracker.get_slot("primary_stakeholders")

        if not primary_stakeholders:
            dispatcher.utter_message(
                text="It seems you haven't provided the primary stakeholders yet. Could you please list the primary stakeholders involved in the project?"
            )
            return []
        validation_prompt = f"Evaluate the following primary stakeholders (those that are primarily affected by the project) for the project: {primary_stakeholders}\n\nAre those stakeholders relevant, clear, and well-defined based on the problem desciption and goal?\nGoal: '{project_goal}'\nProblem description: '{project_problem}'\n\nProvide any suggestions to extend the list, if necessary."

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=350,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"Thank you! The provided feedback on your inputs: {llm_feedback}"
            )
            return [
                SlotSet("proposed_primary_stakeholders", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your primary stakeholders. Please try again."
            )
            return []

class ActionAdjustPrimaryStakeholders(Action):
    def name(self) -> Text:
        return "action_adjust_primary_stakeholders"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]: 
        confirmation = tracker.get_slot("primary_stakeholders_confirmation")
        proposed_primary_stakeholders = tracker.get_slot("proposed_primary_stakeholders")
        current_primary_stakeholders = tracker.get_slot("primary_stakeholders")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed primary stakeholders yet. Could you please provide again the primary stakeholders of your project?"
            )
            return [
                SlotSet("primary_stakeholders", None)
            ]       
        validation_prompt = f"Based on the user's confirmation and the proposed list of primary stakeholders, can you update the existing primary stakeholders of the system?\n\nConfirmation: '{confirmation}'\nProposed primary stakeholders: '{proposed_primary_stakeholders}'\nExisting primary stakeholders: '{current_primary_stakeholders}'"

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final list of primary stakeholders is: {llm_feedback}"
            )
            return [
                SlotSet("primary_stakeholders", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error adjusting the list of primary stakeholders. Please try again."
            )
            return []

class ActionValidateSecondaryStakeholders(Action):
    def name(self) -> Text:
        return "action_validate_secondary_stakeholders"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        project_problem = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")
        secondary_stakeholders = tracker.get_slot("secondary_stakeholders")

        if not secondary_stakeholders:
            dispatcher.utter_message(
                text="It seems you haven't provided the secondary stakeholders yet. Could you please list the primary stakeholders involved in the project?"
            )
            return [
                SlotSet("secondary_stakeholders", None)
            ]   
        validation_prompt = f"Evaluate the following secondary stakeholders (those that are indirectly affected by the project) for the project:\n{secondary_stakeholders}\nAre those secondary stakeholders relevant, clear, and well-defined based on the problem desciption and goal?\nGoal: '{project_goal}'\nProblem description: '{project_problem}'\n\nSuggest improvements if needed."

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=350,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The provided feedback on your secondary stakeholders of the project is: {llm_feedback}"
            )
            return [
                SlotSet("proposed_secondary_stakeholders", llm_feedback)
            ]
        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your secondary stakeholders. Please try again."
            )
            return []

class ActionAdjustSecondaryStakeholders(Action):
    def name(self) -> Text:
        return "action_adjust_secondary_stakeholders"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]: 
        confirmation = tracker.get_slot("secondary_stakeholders_confirmation")
        proposed_secondary_stakeholders = tracker.get_slot("proposed_secondary_stakeholders")
        current_secondary_stakeholders = tracker.get_slot("secondary_stakeholders")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed secondary stakeholders yet. Could you please provide again the secondary stakeholders of your project?"
            )
            return [
                SlotSet("secondary_stakeholders", None)
            ]       
        validation_prompt = f"Based on the user's confirmation and the proposed list of secondary stakeholders, can you update the existing secondary stakeholders of the system?\n\nConfirmation: '{confirmation}'\nProposed secondary stakeholders: '{proposed_secondary_stakeholders}'\nExisting secondary stakeholders: '{current_secondary_stakeholders}'"

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final list of secondary stakeholders is: {llm_feedback}"
            )
            return [
                SlotSet("secondary_stakeholders", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error adjusting the list of secondary stakeholders. Please try again."
            )
            return []

class ActionElicitFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_elicit_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        project_problem = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")
        current_functional_requirements = tracker.get_slot("functional_requirements")

        if not current_functional_requirements:
            dispatcher.utter_message(text="It seems you haven't provided any functionality yet. Could you please list any ideas regarding the functionality of the project?")
            return []
        
        validation_prompt = f"Based on the following problem description, project goal and existing functionality, do you have any suggestions to extend the Functional Requirements of the system?\n\nProblem: '{project_problem}'\nGoal: '{project_goal}'\nExisting Functionality: '{current_functional_requirements}'"       

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The provided feedback on your inputs: {llm_feedback}"
            )
            return [
                SlotSet("proposed_functional_requirements", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error eliciting functional requirements. Please try again."
            )
            return []

class ActionAddProposedFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_add_proposed_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        confirmation = tracker.get_slot("functional_requirements_confirmation")
        proposed_functionality = tracker.get_slot("proposed_functional_requirements")
        current_functional_requirements = tracker.get_slot("functional_requirements")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed functionality yet. Could you please provide your feedback regarding the proposed functionality of the project?"
            )
            return []
        
        validation_prompt = f"Based on the user's confirmation and the proposed functionality, can you update the existing functionality of the system? Please provide the functionality as a list of Functional Requirements, e.g. 'The system shall...'\n\nConfirmation: '{confirmation}'\nProposed Functionality: '{proposed_functionality}'\nExisting Functionality: '{current_functional_requirements}'"       

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final list of functional requirements is: {llm_feedback}"
            )
            return [
                SlotSet("functional_requirements", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error eliciting functional requirements. Please try again."
            )
            return []

class ActionValidateFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_validate_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Retrieve the user's input from the slot
        functional_requirements = tracker.get_slot("functional_requirements")

        # Use the LLM to validate the functional requirement
        validation_prompt = f"Evaluate the following functional requirements: '{functional_requirements}'. Are those functional requirements clear, specific, and actionable?"
        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            # If the feedback indicates the requirement is unclear, ask the user for clarification
            if "vague" in llm_feedback.lower() or "unclear" in llm_feedback.lower() or "improvement" in llm_feedback.lower():
                dispatcher.utter_message(
                    text=f"The LLM provided feedback: {llm_feedback}"
                )
                dispatcher.utter_message(
                    text="Could you please clarify or provide a more detailed description of the functionality?"
                )
                return [
                    SlotSet("functional_requirements", None)
                ]
            else:
                dispatcher.utter_message(
                    text=f"Thank you! The functional requirements are clear: {functional_requirements}"
                )
                return [
                    SlotSet("functional_requirements", functional_requirements)
                ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your functional requirement. Please try again."
            )
            return []

class ActionElicitNonFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_elicit_non_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        project_problem = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")
        current_functional_requirements = tracker.get_slot("functional_requirements")
        current_non_functional_requirements = tracker.get_slot("non_functional_requirements")

        if not current_non_functional_requirements:
            dispatcher.utter_message(
                text="It seems you haven't provided any quality attributes yet. Could you please list any ideas regarding the quality attributes of the project?"
            )
            return []
        
        validation_prompt = f"Based on the following problem description, project goal, existing functionality, and non-functional requirements, do you have any suggestions to extend the quality atrributes of the system?\n\nProblem: '{project_problem}'\nGoal: '{project_goal}'\nExisting Functionality: '{current_functional_requirements}'\nExisting Quality Attributes:'{current_non_functional_requirements}'"       

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The provided feedback on your inputs: {llm_feedback}"
            )
            dispatcher.utter_message(
                text="Could you please verify which suggections, if some, you want to include in the final list of non-functional requirements?"
            )
            return [
                SlotSet("proposed_non_functional_requirements", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error eliciting functional requirements. Please try again."
            )
            return []

class ActionAddProposedNonFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_add_proposed_non_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        confirmation = tracker.get_slot("non_functional_requirements_confirmation")
        proposed_non_functional_requirements = tracker.get_slot("proposed_non_functional_requirements")
        current_non_functional_requirements = tracker.get_slot("non_functional_requirements")

        if not confirmation:
            dispatcher.utter_message(
                text="It seems you haven't provided any feedback on the proposed quality attributes yet. Could you please provide your feedback regarding the proposed non-functional requirements of the project?"
            )
            return []
        
        validation_prompt = f"Based on the following user's confirmation and the proposed quality attributes, can you update the existing non-functional requirements of the system? Please provide the output as a list of non-Functional Requirements, e.g. 'The system shall...'.\n\nConfirmation: '{confirmation}'\nProposed Quality Attributes: '{proposed_non_functional_requirements}'\nExisting non-Functional Requirements: '{current_non_functional_requirements}'"       

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The final list of non-functional requirements is: {llm_feedback}"
            )
            return [
                SlotSet("non_functional_requirements", llm_feedback)
            ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error eliciting functional requirements. Please try again."
            )
            return []

class ActionValidateNonFunctionalRequirements(Action):
    def name(self) -> Text:
        return "action_validate_non_functional_requirements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # Retrieve the user's input from the slot
        functional_requirements = tracker.get_slot("functional_requirements")
        non_functional_requirements = tracker.get_slot("non_functional_requirements")

        if not non_functional_requirements:
            dispatcher.utter_message(
                text="It seems like you haven't provided any non-functional requirements. Can you please describe the quality attributes that are important in your project?"
            )
            return [
                SlotSet("non_functional_requirements", None)
            ]

        # Use the LLM to validate the functional requirement
        validation_prompt = f"Evaluate the following non-functional requirements: '{non_functional_requirements}', based on those functional requirements: '{functional_requirements}'. Are those non-functional requirements clear, specific, relevant, and actionable? Suggest improvements if needed."

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            # If the feedback indicates the requirement is unclear, ask the user for clarification
            if "vague" in llm_feedback.lower() or "unclear" in llm_feedback.lower() or "improvement" in llm_feedback.lower():
                dispatcher.utter_message(
                    text=f"The provided feedback on your inputs: {llm_feedback}"
                )
                dispatcher.utter_message(
                    text="Could you please clarify or provide a more detailed description of the quality attributes?"
                )
                return [
                    SlotSet("non_functional_requirements", None)
                ]
            else:
                dispatcher.utter_message(
                    text=f"Thank you! The non-functional requirements are clear: {non_functional_requirements}"
                )
                return [
                    SlotSet("non_functional_requirements", non_functional_requirements)
                ]

        except Exception:
            dispatcher.utter_message(
                text="There was an error validating your non-functional requirement. Please try again."
            )
            return []

class ActionSummarizeInterview(Action):
    def name(self) -> Text:
        return "action_summarize_interview"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # Retrieve the user's input from the slot
        problem_description = tracker.get_slot("problem")
        project_goal = tracker.get_slot("goal")
        primary_stakeholders = tracker.get_slot("primary_stakeholders")
        secondary_stakeholders = tracker.get_slot("secondary_stakeholders")
        functional_requirements = tracker.get_slot("functional_requirements")
        non_functional_requirements = tracker.get_slot("non_functional_requirements")

        # Use the LLM to validate the functional requirement
        validation_prompt = f"Summarize the following information:\n\n Problem Description: '{problem_description}'\n Project Goal: '{project_goal}'\n Primary Stakeholders: '{primary_stakeholders}'\n Secondary Stakeholders: '{secondary_stakeholders}'\n Functional Requirements: '{functional_requirements}'\n Non-Functional Requirements: '{non_functional_requirements}'"

        try:
            # Add OpenAI API key here [ openai.api_key = ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Requirements Engineer."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            # Extract the LLM's feedback from the response
            llm_feedback = response['choices'][0]['message']['content'].strip()

            dispatcher.utter_message(
                text=f"The summarization of the interview: {llm_feedback}"
            )
            return[]

        except Exception:
            dispatcher.utter_message(
                text="There was an error summarizing the interview. Please try again."
            )
            return []