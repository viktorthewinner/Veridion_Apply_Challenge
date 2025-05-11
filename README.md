<h2>Task
Build a robust company classifier for a new insurance taxonomy.</h2>

<h3>Objectives</h3>
Accept a list of companies with associated data:
– Company Description
– Business Tags
– Sector, Category, Niche Classification
Receive a static taxonomy (a list of labels) relevant to the insurance industry.
Build a solution that accurately classifies these companies, and any similar ones, into one or more labels from the static taxonomy.
Present your results and demonstrate effectiveness.

<h3>Guidelines</h3>

Since this is an unsolved problem without a predefined ground truth, you’ll need to validate your classifier’s performance through your own methods.

- Analyze strengths and weaknesses:
  Explain where your solution excels and where it may need improvement.<br>
  Discuss scalability and how your solution performs with large datasets.<br>
  Reflect on any assumptions made and unknown factors that could impact your solution.<br>

- Ensure your solution truly addresses the problem
  Focus on solving the actual problem, not just implementing complex algorithms. Using embeddings, zero-shot models, TF-IDF, clustering, or other techniques is meaningless if companies are misclassified due to a flawed approach. A well-designed solution is more important than an impressive algorithm.<br>
  Your evaluation should demonstrate that your solution effectively addresses the problem. Simply plotting similarity scores or reporting F1 and accuracy metrics without meaningful validation only measures alignment with your own heuristic, not real-world effectiveness.<br>
  Take the time to deeply understand the problem before writing code. Even the most sophisticated solution is ineffective if it solves the wrong problem. Misalignment in problem definition leads to incorrect conclusions and wasted effort.

- Provide insights into your problem-solving process:
  Why you did what you did, what other paths you considered, and especially why you chose not to pursue them.

- At Veridion, we run similar algorithms on billions of records. While your solution doesn’t need to scale to that level, it would be impressive if it does. For now, however, what matters most is your approach to solving the problem—if your solution is exceptional for the given dataset, we trust that you can scale it effectively using the right tools.

<h3>Resources</h3>
If you’re ready to jump into the problem, please start with the following files:

- List of Companies: company_list
- Insurance Taxonomy: insurance_taxonomy

<h3>Expected Deliverables</h3>

- Solution explanation / presentation
  Provide an explanation or presentation of your solution and results. You have total creative freedom here—feel free to impress with your thinking process, the paths you took or decided not to take, the reasoning behind your decisions and what led to your approach.

- Annotated Input List
  Return the input list with a new column titled “insurance_label” where you have correctly classified each company into one or more labels from the insurance taxonomy.

- Code and Logic
  Include the code that enabled you to achieve this classification for the provided list, and ideally, for any list of any size.

Submit your project
When you’re finished with the challenge, please submit the link to your Github project below.

<br>
<h1>DISCLAIMER: I HAVE 2 METHODS</h1>

- Main method:
  I am using a pre-trained LLM, I am the going to fine-tune (manually/supervised).<br>
  The plan is to zero-shot via embeddings a small size of the data. Then I verify the ground truth. After that, the model is trained on the training data.<br>

  Pro points:
  -> I could have better answers
  -> supervised learning (I can check early mistakes)

  Con points:
  -> I am not going full auto
  -> more time spent on developing own model

- Second method:
  The first method I built, checking my knowledge/tutorials/AI. I wrote some strong comments in the code also. I can say as well as you that it is a cheap version, not classy.<br>
  For a small project, it is ok to go with a pre-trained LLM and to get fast answers, but for bigger data, the lack of fine-tuning will affect the performance.

<br><br>

<h2>Another thoughts</h2>

- My first idea was to code my own LLM from scratch, but then I may not been able to finish the project in time. Also I wasn`t sure if this type of data is trainable for LLM or I could learnt it in time manually. Watched some tutorials and I think I may have the theory in place.

- I thought about giving a shot to using a GPT, but then I was not satisfied with my own learning process. For sure it can be done with tokenization of one of the AIs, but an easy solve is not good for developing own ideas.

- I think that with a simple neural network we can have also some results, but the time to train it is not worth in my opinion (lot of time, mediocre result).
