"""Tabelas usadas na normalização de ingredientes.

Tudo aqui está sem acento e em minúsculas. Mudou uma regra? Acrescente antes
um caso em tests/test_ingredients.py. O hash deste arquivo vai para o
manifesto do pipeline, então uma mudança aqui muda a versão da base.
"""

# forma escrita -> unidade canônica
UNITS: dict[str, str] = {}
for _canon, _forms in {
    "xicara": ["xicara", "xicaras", "xic", "chavena", "chavenas"],
    "colher": ["colher", "colheres", "colherinha", "colherinhas", "colh"],
    "copo": ["copo", "copos", "caneca", "canecas"],
    "lata": ["lata", "latas", "latinha", "latinhas"],
    "kg": ["kg", "quilo", "quilos", "kilo", "kilos"],
    "g": ["g", "gr", "grama", "gramas"],
    "ml": ["ml", "mililitro", "mililitros"],
    "l": ["l", "litro", "litros"],
    "pitada": ["pitada", "pitadas"],
    "pacote": ["pacote", "pacotes", "pacotinho", "pacotinhos"],
    "dente": ["dente", "dentes"],
    "fatia": ["fatia", "fatias"],
    "tablete": ["tablete", "tabletes"],
    "vidro": ["vidro", "vidros", "frasco", "frascos", "vidrinho"],
    "garrafa": ["garrafa", "garrafas"],
    "caixa": ["caixa", "caixas", "caixinha", "caixinhas"],
    "envelope": ["envelope", "envelopes", "sache", "saches", "saquinho", "saquinhos"],
    "maco": ["maco", "macos", "molho", "molhos"],
    "ramo": ["ramo", "ramos", "raminho", "raminhos"],
    "folha": ["folha", "folhas"],
    "cabeca": ["cabeca", "cabecas"],
    "punhado": ["punhado", "punhados"],
    "concha": ["concha", "conchas"],
    "pires": ["pires"],
    "prato": ["prato", "pratos"],
    "calice": ["calice", "calices"],
    "dose": ["dose", "doses"],
    "gota": ["gota", "gotas"],
    "pedaco": ["pedaco", "pedacos"],
    "cubo": ["cubo", "cubos"],
    "tigela": ["tigela", "tigelas"],
    "pote": ["pote", "potes"],
    "pouco": ["pouco", "pouquinho", "bocado"],
}.items():
    for _f in _forms:
        UNITS[_f] = _canon

# adjetivos de tamanho que podem vir logo depois da unidade
UNIT_SIZES = frozenset({
    "pequeno", "pequena", "medio", "media", "grande", "cheio", "cheia", "raso", "rasa",
    "bem", "generoso", "generosa",
})

# palavras que descrevem preparo/estado; saem da chave (nunca a primeira palavra)
QUALIFIERS = frozenset({
    "picado", "picada", "picadinho", "picadinha", "maduro", "madura", "derretido", "derretida",
    "natural", "integral", "ralado", "ralada", "grande", "medio", "media", "pequeno", "pequena",
    "fresco", "fresca", "gelado", "gelada", "frio", "fria", "quente", "morno", "morna",
    "cozido", "cozida", "inteiro", "inteira", "batido", "batida", "peneirado", "peneirada",
    "cortado", "cortada", "fatiado", "fatiada", "refinado", "refinada", "fervente",
    "amolecido", "amolecida", "descascado", "descascada", "lavado", "lavada", "sem",
    "temperatura", "ambiente", "bem", "maduras", "firme", "macio", "macia", "dessalgado",
    "desfiado", "desfiada", "triturado", "triturada", "escorrido", "escorrida",
    "fervendo", "torrado", "torrada", "verde", "vermelho", "vermelha", "amarelo", "amarela",
    "preto", "preta", "roxo", "roxa", "rasa", "cheia", "cheio",
    "espremido", "espremida", "seco", "seca", "amanhecido", "amanhecida",
})

# chaves compostas que não perdem palavras na limpeza
PROTECTED = frozenset({
    "fermento em po", "leite em po", "acucar em po", "chocolate em po", "cafe em po",
    "cacau em po", "canela em po",
})

# palavras terminadas em "s" que não são plural
SINGULAR_S = frozenset({
    "pires", "lapis", "gas", "tres", "mais", "brocolis", "ananas", "onibus", "menos",
    "simples", "seis", "minas", "atras", "pais", "cuscus", "chantilis", "ingles", "frances",
})

# sinônimos aplicados à chave já normalizada (frase inteira)
SYNONYMS: dict[str, str] = {
    "maisena": "amido de milho",
    "maizena": "amido de milho",
    "mucarela": "queijo mussarela",
    "mussarela": "queijo mussarela",
    "muzzarela": "queijo mussarela",
    "mozzarella": "queijo mussarela",
    "queijo mucarela": "queijo mussarela",
    "queijo muzzarela": "queijo mussarela",
    "clara": "ovo",
    "gema": "ovo",
    "clara de ovo": "ovo",
    "gema de ovo": "ovo",
    "ovo caipira": "ovo",
    "oleo de soja": "oleo",
    "oleo de cozinha": "oleo",
    "azeite de oliva": "azeite",
    "azeite de oliveira": "azeite",
    "batata inglesa": "batata",
    "pimenta reino": "pimenta do reino",
    "salsinha": "salsa",
    "cheiro verde": "salsa",
    "margarina": "manteiga",
    "acucar cristal": "acucar",
    "farinha": "farinha de trigo",
    "trigo": "farinha de trigo",
    "sumo de limao": "suco de limao",
    "leite de vaca": "leite",
    "queijo de minas": "queijo minas",
    "catchup": "ketchup",
    "po royal": "fermento em po",
    "fermento quimico em po": "fermento em po",
    "fermento quimico": "fermento em po",
    "caldo de limao": "suco de limao",
    "pinga": "cachaca",
    "massa de tomate": "extrato de tomate",
    "cravo da india": "cravo",
    "bicarbonato": "bicarbonato de sodio",
}

# itens que quase toda casa tem; contam como disponíveis se a pessoa quiser
STAPLES = frozenset({"agua", "sal", "acucar", "oleo", "pimenta do reino"})

# palavras que começam uma instrução de preparo (linha em prosa, não ingrediente)
PREP_VERBS = frozenset({
    "mexa", "mexer", "misture", "misturar", "junte", "juntar", "coloque", "colocar", "leve",
    "levar", "bata", "bater", "adicione", "adicionar", "acrescente", "acrescentar", "ferva",
    "ferver", "cozinhe", "cozinhar", "deixe", "deixar", "despeje", "despejar", "retire",
    "retirar", "sirva", "servir", "unte", "untar", "asse", "assar", "frite", "fritar",
    "pique", "picar", "corte", "cortar", "aqueca", "aquecer", "derreta", "derreter",
    "polvilhe", "polvilhar", "reserve", "reservar", "lave", "lavar", "tempere", "temperar",
    "faca", "fazer", "ponha", "por", "refogue", "refogar", "dissolva", "dissolver",
})
