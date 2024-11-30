use num_complex::Complex;

const EXPECT_DATA_1_BITS: [u8; 194] = [
    0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
    1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
    1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
    0, 0,
];
const EXPECT_DATA_1_FREQ: [Complex<f32>; 389] = [
    Complex {
        re: 1.012683,
        im: 0.5974338,
    },
    Complex {
        re: 1.1364393,
        im: 1.1962154,
    },
    Complex {
        re: 0.8910023,
        im: 0.1261396,
    },
    Complex {
        re: 0.8824154,
        im: 0.017695416,
    },
    Complex {
        re: 0.8028728,
        im: 0.22319135,
    },
    Complex {
        re: 0.77853787,
        im: 0.5548746,
    },
    Complex {
        re: 0.518668,
        im: -0.023772236,
    },
    Complex {
        re: 0.71903634,
        im: -0.12706882,
    },
    Complex {
        re: 0.5670911,
        im: 0.04755027,
    },
    Complex {
        re: 0.5971318,
        im: 0.35555035,
    },
    Complex {
        re: 0.49554452,
        im: -0.1596122,
    },
    Complex {
        re: 0.5532715,
        im: -0.2842474,
    },
    Complex {
        re: 0.65214527,
        im: -0.2038767,
    },
    Complex {
        re: 0.8691076,
        im: 0.26521927,
    },
    Complex {
        re: 0.6526708,
        im: -0.3533905,
    },
    Complex {
        re: 0.6537193,
        im: -0.4215571,
    },
    Complex {
        re: 0.73288405,
        im: -0.4587405,
    },
    Complex {
        re: 0.7111466,
        im: 0.1572345,
    },
    Complex {
        re: 0.16106048,
        im: 0.72632605,
    },
    Complex {
        re: 0.3612982,
        im: 1.2779129,
    },
    Complex {
        re: 0.3310968,
        im: 0.62110263,
    },
    Complex {
        re: 0.8937774,
        im: 0.76478136,
    },
    Complex {
        re: 0.5066266,
        im: 0.45177326,
    },
    Complex {
        re: 0.4172786,
        im: 1.1500117,
    },
    Complex {
        re: 0.5949443,
        im: 0.44153103,
    },
    Complex {
        re: 0.9191409,
        im: 0.4142556,
    },
    Complex {
        re: 0.7083911,
        im: 0.49199694,
    },
    Complex {
        re: 0.072275504,
        im: 0.9537414,
    },
    Complex {
        re: -0.5516805,
        im: 1.0241507,
    },
    Complex {
        re: -1.0946971,
        im: 0.69190747,
    },
    Complex {
        re: -0.49668092,
        im: 0.6489281,
    },
    Complex {
        re: -0.018281769,
        im: 1.0712016,
    },
    Complex {
        re: -0.57228875,
        im: 0.5944857,
    },
    Complex {
        re: -0.696632,
        im: 0.2892434,
    },
    Complex {
        re: -1.0498857,
        im: -0.687762,
    },
    Complex {
        re: -0.214552,
        im: -1.0242497,
    },
    Complex {
        re: 0.39677218,
        im: -1.125159,
    },
    Complex {
        re: 1.0183297,
        im: -0.42961794,
    },
    Complex {
        re: 0.6845785,
        im: 0.13266882,
    },
    Complex {
        re: 0.6722922,
        im: 0.8371839,
    },
    Complex {
        re: -0.32265723,
        im: 0.6893581,
    },
    Complex {
        re: -0.54859686,
        im: 0.7165898,
    },
    Complex {
        re: -0.21020858,
        im: 0.7079764,
    },
    Complex {
        re: 0.48378345,
        im: 0.9104355,
    },
    Complex {
        re: -0.11061333,
        im: 0.7605886,
    },
    Complex {
        re: -0.6322564,
        im: 0.6930359,
    },
    Complex {
        re: -0.9967601,
        im: 0.086731896,
    },
    Complex {
        re: -0.92223775,
        im: -0.12371161,
    },
    Complex {
        re: -1.0949553,
        im: -0.032121077,
    },
    Complex {
        re: -0.6892088,
        im: 1.2684884,
    },
    Complex {
        re: 0.4772467,
        im: 0.8862318,
    },
    Complex {
        re: 0.7309705,
        im: 0.9176167,
    },
    Complex {
        re: 0.090937644,
        im: 1.0016972,
    },
    Complex {
        re: -0.16629212,
        im: 1.1252539,
    },
    Complex {
        re: 0.29363945,
        im: 0.9251226,
    },
    Complex {
        re: 0.7237106,
        im: 0.4443991,
    },
    Complex {
        re: 0.71809036,
        im: -0.43966028,
    },
    Complex {
        re: 0.10195897,
        im: -0.8088783,
    },
    Complex {
        re: -0.57912517,
        im: -0.6785348,
    },
    Complex {
        re: -0.7131168,
        im: -0.40485635,
    },
    Complex {
        re: -0.6223805,
        im: -1.0802618,
    },
    Complex {
        re: -0.029667657,
        im: -0.95021194,
    },
    Complex {
        re: -0.6903264,
        im: -0.88183177,
    },
    Complex {
        re: -0.7144856,
        im: -0.54219186,
    },
    Complex {
        re: -0.60648197,
        im: -0.91811967,
    },
    Complex {
        re: 0.18717305,
        im: -0.98146224,
    },
    Complex {
        re: 0.6738866,
        im: -0.9458647,
    },
    Complex {
        re: 1.0421613,
        im: 0.23754087,
    },
    Complex {
        re: 0.8413035,
        im: 0.4287979,
    },
    Complex {
        re: 0.70807534,
        im: 0.86080414,
    },
    Complex {
        re: 0.6506715,
        im: 0.30099303,
    },
    Complex {
        re: 0.8226319,
        im: -0.13645363,
    },
    Complex {
        re: 0.43767795,
        im: -1.134828,
    },
    Complex {
        re: -0.2740138,
        im: -0.6960271,
    },
    Complex {
        re: -0.96114385,
        im: -0.46291927,
    },
    Complex {
        re: -1.0427094,
        im: 0.5192408,
    },
    Complex {
        re: -0.8350607,
        im: -0.20163393,
    },
    Complex {
        re: -0.7145097,
        im: -0.35739255,
    },
    Complex {
        re: -0.92569757,
        im: -0.25751248,
    },
    Complex {
        re: -0.7617529,
        im: 0.48238134,
    },
    Complex {
        re: -0.20133333,
        im: 0.9163692,
    },
    Complex {
        re: 0.43745106,
        im: 1.0146077,
    },
    Complex {
        re: -0.10936087,
        im: 1.1188112,
    },
    Complex {
        re: -0.51828,
        im: 1.012079,
    },
    Complex {
        re: -0.014234099,
        im: 1.2050949,
    },
    Complex {
        re: 0.90841264,
        im: 1.3311417,
    },
    Complex {
        re: 0.044305377,
        im: 0.75554067,
    },
    Complex {
        re: -0.2049004,
        im: 1.211934,
    },
    Complex {
        re: 0.1384997,
        im: 0.7673783,
    },
    Complex {
        re: 0.9484363,
        im: 0.53582287,
    },
    Complex {
        re: 0.62507844,
        im: -0.5385161,
    },
    Complex {
        re: 0.7430091,
        im: -0.447389,
    },
    Complex {
        re: 0.5901357,
        im: -0.5674504,
    },
    Complex {
        re: 0.93446857,
        im: 0.30778855,
    },
    Complex {
        re: 0.20654622,
        im: 0.6895658,
    },
    Complex {
        re: -0.32549262,
        im: 1.1017325,
    },
    Complex {
        re: -0.68156004,
        im: 0.39954457,
    },
    Complex {
        re: -0.81456596,
        im: 0.13048561,
    },
    Complex {
        re: -1.0617584,
        im: 0.4438694,
    },
    Complex {
        re: -0.60712177,
        im: 1.092725,
    },
    Complex {
        re: -0.69705176,
        im: 0.6835244,
    },
    Complex {
        re: -0.9835748,
        im: 0.33913067,
    },
    Complex {
        re: -0.8935438,
        im: 0.43799555,
    },
    Complex {
        re: 0.05510166,
        im: 1.2483394,
    },
    Complex {
        re: 0.81107795,
        im: 0.6560058,
    },
    Complex {
        re: 1.0360643,
        im: 0.17351264,
    },
    Complex {
        re: 0.5252563,
        im: 0.46332383,
    },
    Complex {
        re: 0.18864776,
        im: 0.9436844,
    },
    Complex {
        re: -0.8234261,
        im: 0.50578916,
    },
    Complex {
        re: -0.84441584,
        im: 0.5065851,
    },
    Complex {
        re: -0.66692287,
        im: -0.3436628,
    },
    Complex {
        re: -0.16305241,
        im: -0.5488679,
    },
    Complex {
        re: 0.36630622,
        im: -1.030584,
    },
    Complex {
        re: 0.9178466,
        im: -0.5956289,
    },
    Complex {
        re: 0.3535313,
        im: -1.0857793,
    },
    Complex {
        re: -0.18182394,
        im: -0.80143106,
    },
    Complex {
        re: -0.0126091475,
        im: -0.7648331,
    },
    Complex {
        re: 0.6234798,
        im: -0.6765491,
    },
    Complex {
        re: 0.13260765,
        im: -1.4814311,
    },
    Complex {
        re: -0.359577,
        im: -0.9246196,
    },
    Complex {
        re: -0.022534119,
        im: -0.97010434,
    },
    Complex {
        re: 0.7105307,
        im: -0.65990806,
    },
    Complex {
        re: 0.92141986,
        im: -0.0150584625,
    },
    Complex {
        re: 0.8064915,
        im: 0.5403656,
    },
    Complex {
        re: 0.7954208,
        im: -0.10630139,
    },
    Complex {
        re: 0.7121406,
        im: -0.84273875,
    },
    Complex {
        re: -0.38362604,
        im: -1.2874466,
    },
    Complex {
        re: -0.8354897,
        im: -0.68923885,
    },
    Complex {
        re: -0.24950211,
        im: -0.76441234,
    },
    Complex {
        re: 0.66378546,
        im: -0.73202246,
    },
    Complex {
        re: 0.82191634,
        im: -0.48575518,
    },
    Complex {
        re: 0.8118297,
        im: 0.5424789,
    },
    Complex {
        re: 0.3713328,
        im: 0.8045468,
    },
    Complex {
        re: -0.33651236,
        im: 1.1211307,
    },
    Complex {
        re: -0.7689243,
        im: 0.52821475,
    },
    Complex {
        re: -0.8155484,
        im: -0.14367217,
    },
    Complex {
        re: -0.7392971,
        im: -0.9504196,
    },
    Complex {
        re: -0.17316778,
        im: -0.82123494,
    },
    Complex {
        re: -0.65901107,
        im: -0.75622654,
    },
    Complex {
        re: -0.9990408,
        im: 0.20552349,
    },
    Complex {
        re: -0.48551607,
        im: 0.47506127,
    },
    Complex {
        re: 0.25199926,
        im: 1.1051539,
    },
    Complex {
        re: 0.71753913,
        im: 0.52829957,
    },
    Complex {
        re: 1.0157746,
        im: -0.055296615,
    },
    Complex {
        re: 0.30102038,
        im: -0.9071268,
    },
    Complex {
        re: 0.17607093,
        im: -1.1345094,
    },
    Complex {
        re: 0.29147518,
        im: -0.76807475,
    },
    Complex {
        re: 0.8372372,
        im: -0.3411676,
    },
    Complex {
        re: 0.28413805,
        im: -0.9199351,
    },
    Complex {
        re: -0.18244503,
        im: -1.0110412,
    },
    Complex {
        re: 0.24815732,
        im: -0.87510926,
    },
    Complex {
        re: 0.82805526,
        im: -0.70775455,
    },
    Complex {
        re: 0.16187899,
        im: -1.11719,
    },
    Complex {
        re: -0.35232612,
        im: -0.6510707,
    },
    Complex {
        re: -0.9154047,
        im: -0.49660107,
    },
    Complex {
        re: -0.78854233,
        im: 0.87759656,
    },
    Complex {
        re: -0.06857278,
        im: 1.2268864,
    },
    Complex {
        re: 0.58639485,
        im: 0.94029355,
    },
    Complex {
        re: -0.22367981,
        im: 0.7506331,
    },
    Complex {
        re: -0.8128377,
        im: 0.96276885,
    },
    Complex {
        re: -0.83571255,
        im: -0.133253,
    },
    Complex {
        re: -1.071269,
        im: -0.20098652,
    },
    Complex {
        re: -0.796886,
        im: 0.04092266,
    },
    Complex {
        re: -0.8312827,
        im: 0.71733737,
    },
    Complex {
        re: -0.8050557,
        im: 0.13007328,
    },
    Complex {
        re: -0.6366421,
        im: -0.49565372,
    },
    Complex {
        re: -0.5651241,
        im: -1.1242507,
    },
    Complex {
        re: 0.45034915,
        im: -0.8955256,
    },
    Complex {
        re: 0.87155956,
        im: -0.48387793,
    },
    Complex {
        re: 1.1293192,
        im: -0.1897751,
    },
    Complex {
        re: 0.60638994,
        im: -0.58559704,
    },
    Complex {
        re: 0.328039,
        im: -0.9378638,
    },
    Complex {
        re: -0.53550667,
        im: -0.7310467,
    },
    Complex {
        re: -0.73154163,
        im: 0.2496298,
    },
    Complex {
        re: -0.7027557,
        im: 0.43683225,
    },
    Complex {
        re: -0.24931231,
        im: 1.2491373,
    },
    Complex {
        re: -0.8017372,
        im: 0.43381688,
    },
    Complex {
        re: -1.0641855,
        im: 0.22558174,
    },
    Complex {
        re: -0.7192029,
        im: 0.54860634,
    },
    Complex {
        re: -0.20986046,
        im: 0.79169816,
    },
    Complex {
        re: -0.58807343,
        im: 0.67466223,
    },
    Complex {
        re: -0.96997535,
        im: 0.31223723,
    },
    Complex {
        re: -1.0005617,
        im: -0.8610017,
    },
    Complex {
        re: -0.38544533,
        im: -0.8260222,
    },
    Complex {
        re: 0.42698574,
        im: -0.8799439,
    },
    Complex {
        re: 0.9846565,
        im: -0.587013,
    },
    Complex {
        re: 0.34393656,
        im: -1.0708443,
    },
    Complex {
        re: -0.45974407,
        im: -0.7851607,
    },
    Complex {
        re: -0.99676394,
        im: -0.31529155,
    },
    Complex {
        re: -0.8574474,
        im: 0.525486,
    },
    Complex {
        re: -0.46602225,
        im: 0.73760164,
    },
    Complex {
        re: 0.6419801,
        im: 0.93641764,
    },
    Complex {
        re: 1.1403424,
        im: -0.029970631,
    },
    Complex {
        re: 0.91456723,
        im: -0.107142225,
    },
    Complex {
        re: 0.95748985,
        im: 0.12703142,
    },
    Complex {
        re: 0.8974865,
        im: 0.6307369,
    },
    Complex {
        re: 0.7618505,
        im: -0.027211094,
    },
    Complex {
        re: 0.85060585,
        im: -0.58441347,
    },
    Complex {
        re: 0.8576482,
        im: -0.28152925,
    },
    Complex {
        re: 0.7872985,
        im: 0.6095902,
    },
    Complex {
        re: -0.13039112,
        im: 0.95196164,
    },
    Complex {
        re: -0.5527334,
        im: 1.1264691,
    },
    Complex {
        re: -0.8988134,
        im: 0.15979497,
    },
    Complex {
        re: -0.93048763,
        im: 0.073891,
    },
    Complex {
        re: -0.80050546,
        im: 0.15287836,
    },
    Complex {
        re: -0.49469563,
        im: 0.8614643,
    },
    Complex {
        re: 0.13358425,
        im: 0.847896,
    },
    Complex {
        re: 0.88901883,
        im: 0.34706512,
    },
    Complex {
        re: 0.67029625,
        im: -0.52346706,
    },
    Complex {
        re: 0.7634218,
        im: -0.74492466,
    },
    Complex {
        re: 0.7899925,
        im: -0.851322,
    },
    Complex {
        re: 0.9708669,
        im: 0.06748385,
    },
    Complex {
        re: 0.7426081,
        im: 0.80158144,
    },
    Complex {
        re: -0.08878782,
        im: 1.1821212,
    },
    Complex {
        re: -0.9741538,
        im: 0.71818495,
    },
    Complex {
        re: -0.91408014,
        im: 0.33774394,
    },
    Complex {
        re: -0.5379567,
        im: 0.57380164,
    },
    Complex {
        re: 0.01961594,
        im: 1.233602,
    },
    Complex {
        re: 0.7846931,
        im: 0.5731107,
    },
    Complex {
        re: 0.85240483,
        im: 0.003212392,
    },
    Complex {
        re: 0.6591547,
        im: -0.7941162,
    },
    Complex {
        re: -0.19449931,
        im: -0.7468863,
    },
    Complex {
        re: -0.68356603,
        im: -0.38148502,
    },
    Complex {
        re: -1.0124792,
        im: 0.12529087,
    },
    Complex {
        re: -0.7727306,
        im: -0.5142596,
    },
    Complex {
        re: -0.840759,
        im: -0.5918646,
    },
    Complex {
        re: -0.76493675,
        im: -0.30395284,
    },
    Complex {
        re: -1.0502625,
        im: 0.46222317,
    },
    Complex {
        re: -0.99572885,
        im: -0.23779556,
    },
    Complex {
        re: -0.5104168,
        im: -0.9248022,
    },
    Complex {
        re: 0.006312267,
        im: -1.1339291,
    },
    Complex {
        re: 0.6397048,
        im: -0.8811573,
    },
    Complex {
        re: -0.22689795,
        im: -1.1261823,
    },
    Complex {
        re: -0.52719426,
        im: -0.76069117,
    },
    Complex {
        re: 0.006291283,
        im: -1.0484443,
    },
    Complex {
        re: 0.4631715,
        im: -1.0284345,
    },
    Complex {
        re: -0.3948469,
        im: -0.986392,
    },
    Complex {
        re: -0.58021426,
        im: -0.49068916,
    },
    Complex {
        re: -0.18153767,
        im: -1.0348682,
    },
    Complex {
        re: 0.5341958,
        im: -0.8334041,
    },
    Complex {
        re: 0.7114807,
        im: -0.23710169,
    },
    Complex {
        re: 0.76315147,
        im: 0.31282663,
    },
    Complex {
        re: 0.81309617,
        im: -0.20582192,
    },
    Complex {
        re: 0.6453499,
        im: -0.7768598,
    },
    Complex {
        re: -0.408825,
        im: -1.0364227,
    },
    Complex {
        re: -0.97247595,
        im: -0.35861713,
    },
    Complex {
        re: -0.7847669,
        im: 0.35229978,
    },
    Complex {
        re: -0.6105397,
        im: 0.9815612,
    },
    Complex {
        re: -0.73263454,
        im: 0.43941838,
    },
    Complex {
        re: -1.0254157,
        im: 0.1776741,
    },
    Complex {
        re: -0.5667323,
        im: -0.6315905,
    },
    Complex {
        re: -0.14944571,
        im: -1.1025529,
    },
    Complex {
        re: 0.7657737,
        im: -0.70250344,
    },
    Complex {
        re: 1.3009455,
        im: -0.28157383,
    },
    Complex {
        re: 0.6121133,
        im: -0.93869936,
    },
    Complex {
        re: 0.13764665,
        im: -0.966039,
    },
    Complex {
        re: 0.5970826,
        im: -0.9092641,
    },
    Complex {
        re: 0.9836179,
        im: -0.6682471,
    },
    Complex {
        re: 0.3278241,
        im: -0.888997,
    },
    Complex {
        re: -0.19339383,
        im: -0.55082893,
    },
    Complex {
        re: -0.6518157,
        im: -0.4350774,
    },
    Complex {
        re: -0.74625915,
        im: 0.25811976,
    },
    Complex {
        re: -0.39234787,
        im: 0.9199515,
    },
    Complex {
        re: 0.49324602,
        im: 0.8695397,
    },
    Complex {
        re: 0.8144212,
        im: 0.27140388,
    },
    Complex {
        re: 1.154666,
        im: 0.14454639,
    },
    Complex {
        re: 1.0698868,
        im: -0.01170371,
    },
    Complex {
        re: 0.6842046,
        im: 0.6210314,
    },
    Complex {
        re: 0.7332075,
        im: 0.098713204,
    },
    Complex {
        re: 1.2394376,
        im: -0.2538902,
    },
    Complex {
        re: 1.2109427,
        im: -0.27151263,
    },
    Complex {
        re: 0.8131491,
        im: 0.5237976,
    },
    Complex {
        re: 0.77452755,
        im: 0.053256888,
    },
    Complex {
        re: 0.7611602,
        im: -0.7442924,
    },
    Complex {
        re: -0.07654403,
        im: -0.9456529,
    },
    Complex {
        re: -0.8366679,
        im: -0.76722854,
    },
    Complex {
        re: -0.45456004,
        im: -0.90287167,
    },
    Complex {
        re: 0.5363878,
        im: -0.75730497,
    },
    Complex {
        re: -0.2745634,
        im: -1.0101048,
    },
    Complex {
        re: -0.58620965,
        im: -0.5941501,
    },
    Complex {
        re: -0.40852398,
        im: -1.075504,
    },
    Complex {
        re: 0.61021596,
        im: -0.7501822,
    },
    Complex {
        re: 0.8499469,
        im: -0.4623419,
    },
    Complex {
        re: 0.99266344,
        im: 0.21089853,
    },
    Complex {
        re: 1.0474644,
        im: -0.55215156,
    },
    Complex {
        re: 0.46378762,
        im: -0.9385504,
    },
    Complex {
        re: -0.72104543,
        im: -0.99460536,
    },
    Complex {
        re: -0.83752877,
        im: -0.16434331,
    },
    Complex {
        re: -0.81412303,
        im: 0.40110508,
    },
    Complex {
        re: -0.40325114,
        im: 1.1111664,
    },
    Complex {
        re: -0.58922213,
        im: 0.53251106,
    },
    Complex {
        re: -0.8306958,
        im: 0.052835096,
    },
    Complex {
        re: -0.7155122,
        im: -0.56212485,
    },
    Complex {
        re: -0.34998327,
        im: -0.62378204,
    },
    Complex {
        re: -0.70691085,
        im: -0.8925841,
    },
    Complex {
        re: -1.0304954,
        im: 0.43880743,
    },
    Complex {
        re: -0.5457571,
        im: 0.70024425,
    },
    Complex {
        re: -0.089298815,
        im: 1.0183247,
    },
    Complex {
        re: -0.4724672,
        im: 0.90690595,
    },
    Complex {
        re: -0.6292083,
        im: 0.7537007,
    },
    Complex {
        re: -0.37123284,
        im: 0.916818,
    },
    Complex {
        re: 0.08537888,
        im: 1.0704253,
    },
    Complex {
        re: -0.36391464,
        im: 0.58382374,
    },
    Complex {
        re: -0.39366183,
        im: 0.8394286,
    },
    Complex {
        re: -0.30448505,
        im: 0.8040703,
    },
    Complex {
        re: 0.5507018,
        im: 1.2525926,
    },
    Complex {
        re: -0.006471966,
        im: 0.6438851,
    },
    Complex {
        re: -0.73321086,
        im: 0.81161684,
    },
    Complex {
        re: -1.0242617,
        im: -0.2750092,
    },
    Complex {
        re: -0.8687356,
        im: -0.42835423,
    },
    Complex {
        re: -0.9814946,
        im: -0.13791692,
    },
    Complex {
        re: -0.49353883,
        im: 0.84229267,
    },
    Complex {
        re: 0.119382665,
        im: 0.7709284,
    },
    Complex {
        re: 0.7707643,
        im: 0.8983267,
    },
    Complex {
        re: 0.04228767,
        im: 0.8165814,
    },
    Complex {
        re: -0.80714947,
        im: 1.1628246,
    },
    Complex {
        re: -0.96748954,
        im: 0.21388596,
    },
    Complex {
        re: -0.7802033,
        im: -0.15206075,
    },
    Complex {
        re: -0.98532337,
        im: 0.072685905,
    },
    Complex {
        re: -0.28164607,
        im: 0.70266765,
    },
    Complex {
        re: 0.3307799,
        im: 0.77119434,
    },
    Complex {
        re: 1.0920382,
        im: 0.27145556,
    },
    Complex {
        re: 0.83869326,
        im: -0.68335485,
    },
    Complex {
        re: 0.3143252,
        im: -0.59638196,
    },
    Complex {
        re: -0.5931242,
        im: -0.98455906,
    },
    Complex {
        re: -1.0810715,
        im: 0.073766455,
    },
    Complex {
        re: -0.7249425,
        im: 0.50058657,
    },
    Complex {
        re: -0.3625338,
        im: 1.2924393,
    },
    Complex {
        re: -0.7500364,
        im: 0.48506847,
    },
    Complex {
        re: -1.1837785,
        im: -0.2057063,
    },
    Complex {
        re: -0.79151314,
        im: -0.778222,
    },
    Complex {
        re: -0.39309657,
        im: -0.86885333,
    },
    Complex {
        re: -0.97043914,
        im: -0.57943404,
    },
    Complex {
        re: -1.1164143,
        im: 0.07536676,
    },
    Complex {
        re: -0.6900578,
        im: -0.45418042,
    },
    Complex {
        re: -0.67675,
        im: -0.8093763,
    },
    Complex {
        re: -0.9199992,
        im: -0.56441516,
    },
    Complex {
        re: -0.9563201,
        im: 0.017556375,
    },
    Complex {
        re: -0.85840094,
        im: -0.4539743,
    },
    Complex {
        re: -0.4272233,
        im: -0.7560755,
    },
    Complex {
        re: 0.29848456,
        im: -0.917573,
    },
    Complex {
        re: 1.0919331,
        im: -0.30468196,
    },
    Complex {
        re: 0.77241904,
        im: 0.07360959,
    },
    Complex {
        re: 0.8281494,
        im: 0.7306544,
    },
    Complex {
        re: 0.6732324,
        im: 0.00749118,
    },
    Complex {
        re: 1.0296997,
        im: -0.5517301,
    },
    Complex {
        re: -0.105745,
        im: -1.145556,
    },
    Complex {
        re: -0.6950484,
        im: -0.52247983,
    },
    Complex {
        re: -0.9489649,
        im: -0.122323714,
    },
    Complex {
        re: -0.58253735,
        im: 1.0041162,
    },
    Complex {
        re: 0.14939584,
        im: 0.8728896,
    },
    Complex {
        re: 0.50560707,
        im: 1.0880181,
    },
    Complex {
        re: 0.18581721,
        im: 0.8558779,
    },
    Complex {
        re: -0.25439364,
        im: 1.2258695,
    },
    Complex {
        re: 0.16302662,
        im: 0.5207579,
    },
    Complex {
        re: 0.6675754,
        im: 0.92722887,
    },
    Complex {
        re: 0.381894,
        im: 0.709807,
    },
    Complex {
        re: 0.03480566,
        im: 1.1716058,
    },
    Complex {
        re: 0.4043256,
        im: 0.72792965,
    },
    Complex {
        re: 0.89006716,
        im: 0.5794161,
    },
    Complex {
        re: 0.95940745,
        im: -0.5793908,
    },
    Complex {
        re: 0.34330913,
        im: -0.8503396,
    },
    Complex {
        re: -0.67838246,
        im: -0.7929445,
    },
    Complex {
        re: -0.82324106,
        im: -0.40943125,
    },
    Complex {
        re: -0.49273774,
        im: -0.65027225,
    },
    Complex {
        re: 0.18370254,
        im: -0.891523,
    },
    Complex {
        re: 0.69410044,
        im: -0.65451944,
    },
    Complex {
        re: 0.97813773,
        im: 0.14108072,
    },
    Complex {
        re: 0.514149,
        im: 0.70030016,
    },
    Complex {
        re: 0.5350925,
        im: 0.95128673,
    },
    Complex {
        re: 0.8052475,
        im: 0.39704865,
    },
    Complex {
        re: 1.242281,
        im: 0.00041245544,
    },
    Complex {
        re: 0.36753142,
        im: -1.0603662,
    },
    Complex {
        re: -0.13171962,
        im: -0.91257226,
    },
    Complex {
        re: -1.0067092,
        im: -0.73088586,
    },
    Complex {
        re: -1.1182305,
        im: 0.09520594,
    },
    Complex {
        re: -0.78105664,
        im: -0.14943029,
    },
    Complex {
        re: -0.24183783,
        im: -0.87140554,
    },
    Complex {
        re: 0.27257583,
        im: -1.2135541,
    },
    Complex {
        re: 0.95441484,
        im: -0.3957555,
    },
    Complex {
        re: 0.6134959,
        im: 0.16858552,
    },
    Complex {
        re: 0.84673804,
        im: 0.67279625,
    },
    Complex {
        re: 0.58045524,
        im: 0.0061426787,
    },
    Complex {
        re: 0.5109923,
        im: -0.4035909,
    },
    Complex {
        re: -0.017788252,
        im: -0.499171,
    },
    Complex {
        re: 0.06876613,
        im: -0.11021158,
    },
    Complex {
        re: -0.13814615,
        im: -0.13578404,
    },
    Complex {
        re: 0.08512126,
        im: -0.1309367,
    },
    Complex {
        re: -0.41817972,
        im: 0.029939989,
    },
];
