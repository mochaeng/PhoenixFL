from scipy.stats import mannwhitneyu

accuracies_centralized = [
    0.9950199723243713,
    0.995199978351593,
    0.9952600002288818,
    0.9950500130653381,
    0.9950199723243713,
    0.9950600266456604,
    0.9944400191307068,
    0.994920015335083,
    0.9951599836349487,
    0.9955599904060364,
    0.9950399994850159,
    0.9956099987030029,
    0.995199978351593,
    0.9954299926757812,
    0.995169997215271,
    0.9954400062561035,
    0.9957299828529358,
    0.9956499934196472,
    0.9950600266456604,
    0.9953500032424927,
    0.9952700138092041,
    0.9953399896621704,
    0.9950500130653381,
    0.994979977607727,
    0.9953799843788147,
    0.9948199987411499,
    0.9954000115394592,
    0.9953200221061707,
    0.9957299828529358,
    0.9951800107955933,
]
f1_score_centralized = [
    0.9379361867904663,
    0.9406234622001648,
    0.9414236545562744,
    0.9399927258491516,
    0.9397386312484741,
    0.9391475915908813,
    0.9316449165344238,
    0.9362289905548096,
    0.9398309588432312,
    0.9454678893089294,
    0.938200831413269,
    0.9462734460830688,
    0.9414348006248474,
    0.9452235102653503,
    0.9416173100471497,
    0.9436341524124146,
    0.9483113288879395,
    0.9473174214363098,
    0.9409090876579285,
    0.9425713419914246,
    0.9430052638053894,
    0.9425966739654541,
    0.9388813972473145,
    0.9400954246520996,
    0.9444311261177063,
    0.9352338314056396,
    0.9448837637901306,
    0.943231463432312,
    0.9476266503334045,
    0.9416040778160095,
]

accuracies_federated = [
    0.9893500208854675,
    0.9883700013160706,
    0.9881899952888489,
    0.9899500012397766,
    0.984499990940094,
    0.9894800186157227,
    0.9840899705886841,
    0.9877899885177612,
    0.9860100150108337,
    0.987339973449707,
    0.9846600294113159,
    0.9884399771690369,
    0.9892299771308899,
    0.9878000020980835,
    0.9863399863243103,
    0.9892500042915344,
    0.9930899739265442,
    0.9864400029182434,
    0.9895600080490112,
    0.9820799827575684,
    0.9812800288200378,
    0.9838899970054626,
    0.9894400238990784,
    0.9889900088310242,
    0.9868299961090088,
    0.9883000254631042,
    0.9879199862480164,
    0.9882100224494934,
    0.98580002784729,
    0.9884300231933594,
]
f1_score_federated = [
    0.8546074628829956,
    0.8363125324249268,
    0.8342921137809753,
    0.8639501333236694,
    0.774840235710144,
    0.8571816086769104,
    0.7598490715026855,
    0.8271762132644653,
    0.7965682744979858,
    0.8278019428253174,
    0.7717940807342529,
    0.8376860618591309,
    0.8522836565971375,
    0.8298465609550476,
    0.8048571944236755,
    0.8525983691215515,
    0.9106658697128296,
    0.8036489486694336,
    0.8580365180969238,
    0.7220843434333801,
    0.7063989639282227,
    0.7535567283630371,
    0.856404721736908,
    0.8481588363647461,
    0.8093791007995605,
    0.8353040814399719,
    0.8282626867294312,
    0.837222158908844,
    0.7936046719551086,
    0.8384304046630859,
]

statistic_f1, pnorm_f1 = mannwhitneyu(
    f1_score_centralized, f1_score_federated, method="auto"
)


print(f"Client-2: F1-score statistic [{statistic_f1}] | p-norm [{pnorm_f1}]")