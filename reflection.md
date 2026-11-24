# Task 6 — Reflection


## 0. The flat line

Three consecutive rows of my ablation table reported the same total: 64.67, 64.67, 64.67. Two of them were labelled "no year window" and "no per-company retrieval" — components I had spent hours building, switched off — and the score did not move at all. For about ten minutes I believed it. I wrote in my notes the sentence this assignment is practically built to produce: *these components do not matter on this corpus.* It is a satisfying sentence. It sounds like insight. It is the kind of thing you write when you have been staring at numbers for hours and want one of them to mean something.

Then I opened `ablation.py`. My "no query parsing" row had been built by switching off query parsing **and** abstention in one step, and every component downstream of it consumes the parsed query: the year window needs a year, the per-company split needs a company, the tie-breakers need both. Once the parse was gone, turning those three knobs changed nothing, because the thing they act on had already been removed. The flat line was not a finding about the corpus. It was a bug in my experiment, wearing the costume of a result — and worse, a *plausible* costume, because "these components don't matter here" is exactly the kind of thing that is sometimes true.

I fixed it with two assertions — every row must differ from the one above by exactly one component, and the last row must be identical to the submitted pipeline — and the table changed, though not much: 64.67 became 64.33. That is the part I keep coming back to. The same flatness, almost the same numbers, and a completely different meaning. Before the fix it was my own wiring lying to me. After it, it was a real finding with a real explanation — the three mechanisms fire on tens of questions and change the final answer on almost none of them, because the answer was already inside the top five. Same shape, different truth, and only the assertions told them apart.

I built a system that answers questions and a harness that says how well, and by the end of it the harness was the more interesting artifact. The most valuable ten minutes were the ones in which I stopped believing it.

## 1. In ten years, who do I want to be?

I want to be the person who asks whether the number is the right number, rather than the person who produces the number.

That sounds like a slogan, so let me make it concrete with the worst thing that happened this week. Variant V4 raised recall from 86.96% to 100%. A perfect score on the retrieval metric. And question q050 — FY2023 revenue for Brightwater Software in EMEA — came back `NOT_FOUND`, because the chunk the retriever was so proud of also contained Brightwater **Energy**'s regional table for the same year, and the generator, quite reasonably, declined to choose between them. My metric said *found*. My system said *I don't know*. Both statements were true. The metric was the one that was wrong.

I do not think I would have written that sentence a year ago. A year ago I would have submitted the 100% and felt good about it, because the number was the assignment and the number was perfect. What changed is not that I became more careful in general — I am not, and the rest of this document is full of evidence — but that I now have a specific memory of a green dashboard lying to me with my own name on the commit.

In ten years the pipeline in this repository will be something a personal agent writes in the time it takes to describe it. What will not be automatic is the disposition to look at a good dashboard and go hunting for the question it is lying about. I want to be that. It is a smaller ambition than "build great systems," and I have made my peace with the smallness, because I have now seen what the large version costs: you get very good at producing numbers and you stop being the kind of person who checks them.

## 2. Which of my current ambitions still matter?

The ones about *mechanism*, not about output.

The largest single jump in my ablation was 19.86 points — 31.33 to 51.19 — and it came from replacing my hand-written tokenizer. Not from a model. Not from an embedding. Not from anything I would have put on a slide. It came from the fact that my tokenizer turned `2025,` into `2025`, so the year in the question and the year in the table header became two different tokens; and from `robotics'` keeping its apostrophe, so the possessive never matched the company name. Nineteen point nine points, from punctuation.

An agent can find that too, if you tell it to look. What I want to keep is the part where I refuse to accept the jump until I know which line caused it — because the temptation, when a number rises by twenty points, is to write "the tokenizer matters" and move on to the next variant. Vague explanations are free now. Being right about the cause is not.

So the ambition that survives is narrow and a bit unfashionable: I want to understand systems at the level of the line that broke. That ambition is not threatened by agents. It is threatened by my own impatience, which is a much older problem and was not caused by anyone's language model.

## 3. Apart from productivity and career achievement, what do I want to be remembered for?

For writing down the things I chose not to fix.

There is a specific one. The corpus prints negative numbers in accounting notation — `Cost of revenue | 2024: (487.8)` — and my `table_to_text` passes it through as printed, which means the sign is lost the moment a model reads it. I could fix that in about four minutes by emitting `-487.8`. I did not, because Task 1.1's grader compares against the text exactly as printed, and that four-minute fix would have cost eight points.

So I knowingly left a defect in a system I am being graded on, and I put it in the error analysis with the reason and the price attached. I am not proud of it. I am also not going to pretend I would have chosen otherwise, because I would not have. What I hope someone would remember is not the clean score, but that the defect sits in the same document as the score, in my own words, without hedging. A person who documents their trade-offs is more useful than a person who appears to have none — and I have stopped believing that the second kind of person exists. Every system I have looked at closely enough has had a `(487.8)` in it somewhere. The difference is only whether it is written down.

## 4. What would I still struggle for, even if an agent made the struggle unnecessary?

Two things, and they are different in kind.

The first is the struggle to be *legible*. Every part of this pipeline is something I can point at and say what it does and what it costs, and that property was not free — it is the reason the ablation exists at all, and the reason a flat row could be diagnosed instead of shrugged at. There is a version of this homework where I hand the whole thing to a model, get 75/75, and learn nothing except that the model is good. I did use a model to write a lot of this code. What I did not delegate was the map: which knob exists, what turning it should change, and what it means when the change does not appear. That map is the thing I would still fight to hold, even if the code around it writes itself.

The second, and the one that actually got to me, is the struggle to be wrong in public. An ablation is a machine for humiliating yourself. You build it to find out which of your ideas was load-bearing, and the honest answer is usually "not the one you liked." V1 is my example. I rewrote the table text so that it read like an answer — `Accounts receivable, net: 492.1` instead of a row of numbers under column headers — because that is obviously better text, by any human standard. Recall fell from 39.13% to 30.43%. The correct chunk for q051 went from rank 43 to rank 74. Making a passage more readable made it less findable, because the retriever was not reading for sense; it was matching tokens, and I had diluted the tokens that were doing the work.

I would still struggle for that. Not because the struggle produces better software — that particular experiment produced a worse number, and I kept the change anyway because V2 recovered the loss through structure — but because the willingness to run the experiment that can destroy your week is the only thing separating a person who knows from a person who believes.

## 5. Which skills, qualities, or experiences still make me valuable?

Three, ranked by how much I trust them.

**Building the measurement, not just the thing.** The pipeline is about 800 lines of parsing and scoring. `ablation.py` is the part I would show someone. It imports the submitted module rather than re-implementing it, so a row of the table cannot drift away from the code that is graded, and it asserts its own integrity before it reports anything. Almost all of the actual learning in this assignment happened inside that harness and not inside the pipeline. Producing an answer is becoming cheap. Producing the thing that tells you your answer is wrong is not, because it requires you to specify in advance what "wrong" would look like — and to accept the answer when it arrives.

**Domain-shaped suspicion.** Knowing that FY2023 figures exist only inside the FY2024 annual report, so a filter of "exactly FY2025" silently deletes the answer. Knowing that Northgate Robotics and Northgate Payments both exist while Northgate Robotic Systems does not, so a prefix match will cheerfully answer a question about a company that is not there. Knowing that "revenue of approximately $2.4 billion" in a call transcript and the precise line in the income statement are the same fact at two resolutions, and that a paraphrase is not a substitute for the table. None of this is knowledge an agent lacks. It is the habit of checking that an agent will not perform unless someone makes it perform, and someone has to know which checks are worth making.

**Knowing when a measurement cannot support a conclusion.** At temperature zero the generator still flips about one question in sixty between runs. So differences below 0.34 points in my table are not interpretable, and I had to write "flat" next to two variants I had hoped would be improvements. That was harder than it should have been. It is also the only reason the rest of the table is worth reading.

## 6. If my value is no longer mainly "what I can do," how do I define who I am?

I do not have a clean answer and I have decided not to manufacture one.

Half of me thinks the answer is: you are what you choose to pay attention to. The other half thinks that is a comfortable sentence, written by someone who is still employable, and that the honest version is closer to *I don't know yet, and I will find out by watching what I do when nobody is grading it.*

What I do have is one piece of evidence, and it surprised me. The most satisfying hour of this assignment was not the run that printed 75.00/75. It was finding the G3 bug. My abstention gate was supposed to reject a question when the corpus never mentions the metric at all. It stripped stopwords first, and "change in working capital" became "change working" — a phrase that appears in no document in the corpus — so the gate rejected a question that was perfectly answerable. Every aggregate number was perfect. Questions kept: 100%. Unanswerable questions rejected: 100%. And the component was broken. I found it by reading the gate's decisions one at a time instead of reading its score.

That hour felt better than the score did. So the honest answer to "who are you" is probably: someone who is moved more by finding a defect than by being told he did well. I would rather know that about myself than have a philosophy.

## 7. What is the one thing I should start doing differently tomorrow?

One thing, small enough that I will actually do it: keep a file called `unfixed.md` in every project, and put in it every defect I knowingly ship, with the reason and the price. I started it in this repository, and the first entry is `(487.8)`: sign lost, kept because the grader wants the printed text, cost — a wrong answer on any corpus that asks about a loss. There are two more entries behind it, both of which are things I would rather not have written down, which is roughly the point.

And a second, which is a habit rather than an artifact: before I trust any number I produced, spend ten minutes trying to break it. Not the system — the number. The three assertions in `ablation.py` are that habit written down: three lines of code that stopped me from publishing a conclusion which was really a bug in my own experiment.

The pipeline answers 60 of 60 and I still do not fully trust it. That distrust is the part of this assignment I want to carry out of the course. The agent will do the work. Someone still has to be the one who checks — and I would like that person to be me, not because I am faster, but because I have now learned, in a fairly literal way, what it costs when nobody is.
