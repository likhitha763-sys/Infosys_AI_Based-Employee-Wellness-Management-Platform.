# Error analysis: bert

## SemEval polarity
Errors: 124 of 675 (107 negative tweets predicted positive, 17 positive tweets predicted negative).
Mentions are replaced with @user. 10 random examples of each kind (seed 42).

### Negative predicted as positive
- `2018-En-04124` @user Hahaaa! Was fuming with that 😞😂
- `2018-En-01850` This happened the last time I was painting
- `2018-En-03346` So close; yet so far! #irony
- `2018-En-04063` @user yessss waiting for an epi is for the birds. it sucks. im waiting for walking dead new season😩
- `2018-En-02337` Mothafuckas wanna adopt the dark, but I was born in it
- `2018-En-03260` #TOKYO\nFear death as much as optimism.
- `2018-En-03295` @user @user Very noble of Trump to stand behind his son and not throw him under the bus. #sarcasm
- `2018-En-01639` To recreate my romantic memories, I'd just put a chip on my shoulder #masterchefau
- `2018-En-01212` TARED tabby abaci rabid each dabs
- `2018-En-01430` @user Macron slips up and has a moment of clarity & common sense...now he is a raging racist. Sounds right. #LiberalLogic

### Positive predicted as negative
- `2018-En-01947` On the bright side ma work pals have treated me too a spotty dotty cookie
- `2018-En-03738` Someone just replied all to an email telling someone else off for sending an email to the whole dept mailing list #ironic 😂
- `2018-En-01926` Worst missing is missing ur ownself u used to be once upon a time. #happy #ambitious#innocent #LifeDreamer
- `2018-En-03909` mum got out of a rlly bad car crash completely not injured and i found a rlly sentimental piece of jewellery i thought i'd lost #blessed
- `2018-En-03957` Been spoilt even got a Kylie lip kit 😆
- `2018-En-04329` @user To Pope\nDisclose Fatima's prophecy. Command from \nGod is very angry.\nPlease disclose the prophecy.\nI am a blessed God's daughter.
- `2018-En-02422` My patient gave me a business card and it's given me so much glee to grind the thing.
- `2018-En-03306` Who also miss glee ? #gleek #glee #GleeForever #DreamsComeTrue #dontstopbelieving #gleecast @user
- `2018-En-01138` Marriage is the mourning after the knot before. #laugh #punoftheday
- `2018-En-01630` @user So bright that kill my heart😂

## ISEAR: true emotion (rows) against the 8 most common predicted labels (columns)
```
pred              neutral  sadness  fear  realization  disappointment  disgust  anger  joy
emotion_original                                                                          
anger                 253       56     3           42              71        8    162    1
disgust               323       33    14           39              37      194     14    2
fear                  249       68   268           77              26        5      6    0
joy                   298       14     6           85              15        0      0  175
sadness               187      377     4           48              61        1      6    3
```
