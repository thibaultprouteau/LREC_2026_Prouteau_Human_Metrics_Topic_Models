// Defining a data model.
class Check{
  isEnglish bool @description(#"
    Indicates whether the word is a correct English word in, or related to philosophy of science.
  "#)
  isAcronym bool @description(#"
    Indicates whether the word is an acronym.
  "#)
  reason string @description(#"
    Provides the reasoning behind the classification of the word.
  "#)
}

// Create a function to extract the resume from a string.
function WordChecker(word: string) -> Check {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client OllamaClient // Set OPENAI_API_KEY to use this client.
  prompt #"
   

    Think step-by-step:
      {{ word }} is a correct English word, in, or related to philosophy of science OR You can answer, for any language one of these two questions : Who is {{ word }} in, or related to philosophy of science? or Where is {{ word }} in, or related to philosophy of science? Then output true. {{ word }} is an acronym.

    Answer: {{ ctx.output_format }}
  "#
}



// Test the function with a sample resume. Open the VSCode playground to run this.
test WordChecker {
  functions [WordChecker]
  args {
    word "run"
  }
}
