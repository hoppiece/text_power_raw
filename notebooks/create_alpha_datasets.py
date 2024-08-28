from datasets import load_dataset, Dataset
from collections import Counter
import numpy as np

# alpha値を指定
alpha = 1.01  # 例: alpha = 1.2

# WikiTextデータセットのロード
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# 単語頻度のカウント
word_counts = Counter()
for sentence in dataset['text']:
    words = sentence.split()
    word_counts.update(words)

# 単語を頻度順にソート
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 単語に対するランクを辞書に変換
word_to_rank = {word: rank + 1 for rank, (word, _) in enumerate(sorted_word_counts)}

# ランクに応じて単語出現頻度を調整
adjusted_text = []

for sentence in dataset['text']:
    new_sentence = []
    words = sentence.split()
    for word in words:
        if word in word_counts:
            rank = word_to_rank[word]
            # 出現頻度を調整
            prob = rank**(1-alpha)
            if np.random.rand() < prob:
                new_sentence.append(word)
            else:
                new_sentence.append(f"[削除, {word}]")
    adjusted_text.append(" ".join(new_sentence))

# 新しいデータセットの作成
new_dataset = Dataset.from_dict({"text": adjusted_text})

# 新しいデータセットの保存
new_dataset.save_to_disk(f"annotated-wikitext-alpha-{alpha}")

"""
= Valkyria [削除, Chronicles] III [削除, =]

[削除, Senjō] [削除, no] Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the [削除, Battlefield] 3 ) , commonly referred to as Valkyria [削除, Chronicles] III [削除, outside] Japan , is a [削除, tactical] [削除, role] @-@ playing [削除, video] [削除, game] developed by Sega and Media.Vision for the [削除, PlayStation] [削除, Portable] . [削除, Released] in [削除, January] 2011 [削除, in] Japan , it is the third game in the Valkyria [削除, series] . Employing the same [削除, fusion] of tactical [削除, and] [削除, real] @-@ time gameplay as its predecessors , the story runs parallel to the first game and [削除, follows] the " Nameless " , a [削除, penal] [削除, military] [削除, unit] serving the [削除, nation] of Gallia during the Second Europan War who perform [削除, secret] black operations and are pitted against the Imperial [削除, unit] " Calamaty Raven " .
The game began [削除, development] in 2010 , [削除, carrying] over a large portion of the work done on Valkyria Chronicles II . While it [削除, retained] the standard [削除, features] of the series , it [削除, also] [削除, underwent] [削除, multiple] adjustments , such as making the game more forgiving for series newcomers . [削除, Character] [削除, designer] [削除, Raita] Honjou and [削除, composer] Hitoshi Sakimoto both returned from previous entries [削除, ,] along with Valkyria Chronicles [削除, II] [削除, director] Takeshi Ozawa . A large team [削除, of] writers handled the script . The game [削除, 's] [削除, opening] theme was sung by May 'n .
It [削除, met] with positive sales in Japan , and [削除, was] [削除, praised] by [削除, both] [削除, Japanese] and western critics . After release , it received downloadable content , [削除, along] with an expanded edition in [削除, November] of that year . It was also [削除, adapted] into [削除, manga] and an original video [削除, animation] series . Due to low sales of [削除, Valkyria] [削除, Chronicles] II , Valkyria Chronicles [削除, III] was not [削除, localized] , but [削除, a] fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure [削除, Revolution] for the [削除, PlayStation] [削除, 4] .

= = Gameplay = =

[削除, As] [削除, with] previous Valkyira Chronicles [削除, games] , [削除, Valkyria] Chronicles III is a tactical role @-@ [削除, playing] [削除, game] where [削除, players] take control of [削除, a] military [削除, unit] and take part in missions against enemy [削除, forces] . Stories are [削除, told] through [削除, comic] book @-@ like panels with animated character [削除, portraits] , with [削除, characters] [削除, speaking] [削除, partially] through voiced speech bubbles and [削除, partially] [削除, through] [削除, unvoiced] [削除, text] . The player [削除, progresses] through a series of [削除, linear] [削除, missions] , [削除, gradually] unlocked as maps that [削除, can] be freely scanned through and [削除, replayed] as they are [削除, unlocked] . The [削除, route] to each story [削除, location] on the map varies [削除, depending] on an individual [削除, player] 's approach : when one option is selected , the other is sealed off to the player . [削除, Outside] missions , the [削除, player] characters rest in a camp , [削除, where] units [削除, can] be customized and character growth [削除, occurs] . [削除, Alongside] the main story missions are character @-@ specific sub [削除, missions] relating [削除, to] different [削除, squad] members . After the game 's completion , additional [削除, episodes] are [削除, unlocked] , [削除, some] of them [削除, having] a higher difficulty than [削除, those] [削除, found] in the rest of the game . There are also love simulation [削除, elements] related to the game 's two main heroines , [削除, although] they take a very minor role .
The game 's [削除, battle] [削除, system] , the BliTZ system , is [削除, carried] over [削除, directly] from Valkyira Chronicles . [削除, During] missions , [削除, players] [削除, select] each unit using a top @-@ [削除, down] perspective of the battlefield map : once [削除, a] [削除, character] is selected , the player moves the character [削除, around] the battlefield in third [削除, @-@] [削除, person] [削除, .] A character can only act once per @-@ [削除, turn] , but characters [削除, can] be granted multiple turns [削除, at] the [削除, expense] of other characters ' turns . Each [削除, character] has a field and [削除, distance] of movement [削除, limited] [削除, by] [削除, their] Action Gauge . Up to nine [削除, characters] can be [削除, assigned] to a single mission . [削除, During] gameplay , characters [削除, will] call out [削除, if] something [削除, happens] to them , such as their health points [削除, (] HP ) getting [削除, low] or being [削除, knocked] out by enemy [削除, attacks] . [削除, Each] character [削除, has] [削除, specific] " Potentials " , [削除, skills] [削除, unique] [削除, to] each character . They are divided [削除, into] " Personal Potential " , which are innate skills [削除, that] remain unaltered [削除, unless] [削除, otherwise] dictated by the story and can either help or impede a [削除, character] , and [削除, "] [削除, Battle] Potentials " , which are grown throughout the game and always grant [削除, boons] to a character . To learn [削除, Battle] Potentials , [削除, each] character has a unique " Masters Table " , a [削除, grid] @-@ based [削除, skill] table that can be used to [削除, acquire] and [削除, link] different skills . Characters also [削除, have] [削除, Special] Abilities that grant them temporary [削除, boosts] on the battlefield : [削除, Kurt] can [削除, activate] " [削除, Direct] Command " and move [削除, around] the [削除, battlefield] without depleting his Action [削除, Point] [削除, gauge] , the character [削除, Reila] [削除, can] [削除, shift] [削除, into] her " Valkyria Form " and become [削除, invincible] , while Imca can target multiple enemy units with her [削除, heavy] [削除, weapon] [削除, .]
Troops are divided [削除, into] five classes : Scouts , Shocktroopers , Engineers , Lancers and [削除, Armored] Soldier . Troopers can switch classes by changing their assigned weapon . Changing class does not greatly affect the [削除, stats] gained while in a previous class . With victory in battle [削除, ,] [削除, experience] points are awarded to the squad , which [削除, are] [削除, distributed] into five [削除, different] attributes shared by the entire squad [削除, ,] a [削除, feature] [削除, differing] from early games ' method of [削除, distributing] to different [削除, unit] types .

= = [削除, Plot] = =

The [削除, game] [削除, takes] place [削除, during] the Second Europan War . [削除, Gallian] Army Squad 422 , [削除, also] [削除, known] as " The [削除, Nameless] " , are a penal military unit composed of criminals , foreign [削除, deserters] , and military offenders whose real names are erased from the records [削除, and] thereon officially [削除, referred] [削除, to] by [削除, numbers] . Ordered by the Gallian military [削除, to] perform the [削除, most] dangerous missions that the Regular [削除, Army] and Militia will [削除, not] do , they are [削除, nevertheless] up to the task , [削除, exemplified] [削除, by] [削除, their] motto , [削除, Altaha] Abilia , [削除, meaning] " [削除, Always] Ready . " The [削除, three] main characters are [削除, No.7] [削除, Kurt] [削除, Irving] , an army officer falsely accused of treason who wishes to redeem himself [削除, ;] [削除, Ace] No.1 [削除, Imca] , a [削除, female] Darcsen heavy weapons [削除, specialist] who seeks [削除, revenge] against the Valkyria [削除, who] [削除, destroyed] her home ; and [削除, No.13] Riela [削除, Marcellis] , a seemingly [削除, jinxed] [削除, young] woman who is unknowingly [削除, a] descendant of the Valkyria . [削除, Together] with their [削除, fellow] squad [削除, members] , these three are [削除, tasked] to [削除, fight] against a mysterious Imperial unit [削除, known] as Calamity Raven , consisting of mostly Darcsen soldiers .
As the [削除, Nameless] officially do not exist , the upper echelons of the [削除, Gallian] [削除, Army] [削除, exploit] the concept of plausible deniability in order to send them [削除, on] [削除, missions] that [削除, would] [削除, otherwise] make Gallia [削除, lose] face in the war . While [削除, at] times this works [削除, to] their advantage , such as a [削除, successful] incursion [削除, into] Imperial territory , other [削除, orders] [削除, cause] [削除, certain] members of the [削除, 422nd] [削除, great] distress . One such member , [削除, Gusurg] , [削除, becomes] so enraged that [削除, he] [削除, abandons] his post and defects into the ranks of Calamity Raven , attached to the ideal of Darcsen independence proposed by their leader , Dahau . At the same time , elements within Gallian Army [削除, Command] move to erase the Nameless [削除, in] order to protect their own [削除, interests] . Hounded by both allies and [削除, enemies] , and combined with the [削除, presence] of a [削除, traitor] within their [削除, ranks] , the 422nd desperately move to [削除, keep] [削除, themselves] alive while at the same time fight to [削除, help] the [削除, Gallian] war [削除, effort] . This [削除, continues] until the Nameless 's [削除, commanding] [削除, officer] , Ramsey Crowe , who had been kept under house arrest , is [削除, escorted] to the capital city of [削除, Randgriz] in order to present [削除, evidence] exonerating the weary soldiers and expose the real traitor , the Gallian General that had accused Kurt of [削除, Treason] .
Partly due to these events , and partly due to the major [削除, losses] in manpower [削除, Gallia] [削除, suffers] [削除, towards] the [削除, end] of the war with the Empire , the Nameless [削除, are] [削除, offered] a [削除, formal] position as a squad in the Gallian Army rather [削除, than] serve as an [削除, anonymous] shadow [削除, force] . [削除, This] is [削除, short] @-@ lived , however , as [削除, following] Maximilian 's defeat , [削除, Dahau] and [削除, Calamity] Raven [削除, move] to activate an ancient Valkyrian super weapon within the Empire , kept secret by their benefactor [削除, .] [削除, Without] the support of [削除, Maximilian] or the chance to prove [削除, themselves] in the war with [削除, Gallia] , it is [削除, Dahau] 's last trump card in creating a new [削除, Darcsen] nation . As an armed [削除, Gallian] force invading the Empire just following the two [削除, nations] ' cease [削除, @-@] fire would certainly wreck [削除, their] [削除, newfound] [削除, peace] , Kurt decides to [削除, once] again [削除, make] his squad the Nameless , [削除, asking] [削除, Crowe] to [削除, list] himself and [削除, all] [削除, under] his [削除, command] [削除, as] killed [削除, @-@] in @-@ action . [削除, Now] [削除, owing] [削除, allegiance] to none other than themselves , the [削除, 422nd] [削除, confronts] Dahau and destroys the [削除, Valkyrian] weapon . [削除, Each] [削除, member] then goes their separate [削除, ways] in order to [削除, begin] their [削除, lives] anew .

= = [削除, Development] [削除, =] =
"""