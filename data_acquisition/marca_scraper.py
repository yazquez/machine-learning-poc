from bs4 import BeautifulSoup
import requests

user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0'
headers = {
    'User-Agent': user_agent
}

# Look at the terms and the robots.txt file
url = "http://www.marca.com/futbol/primera/calendario.html"
soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html5lib')

resultados = []
jornadas = soup('div', 'jornadaCalendario')

# Marca example - Second Part
# Look at the terms and the robots.txt file
for jornada in jornadas:
    # Date and number data
    datos_jornada = jornada.find('div', 'datosJornada')
    nombre_jornada = datos_jornada.find('h2').text
    fecha_jornada = datos_jornada.find('p').text
    # Matches data
    partidos_jornada = jornada.find('ul', 'partidoJornada')

    # Marca example - Third Part
    # Look at the terms and the robots.txt file
    for partido_jornada in partidos_jornada:
        local = ""
        visitante = ""
        try:
            local = partido_jornada.find('span', 'local').text
            visitante = partido_jornada.find('span', 'visitante').text
            resultado = partido_jornada.find('span', 'resultado').text
        except:
            pass

        # Marca example - Fourth Part
        # Look at the terms and the robots.txt file
        if 'Betis' in [local, visitante]:
            partido = u"{0} vs {1}: {2} {3}".format(local, visitante, resultado, fecha_jornada)
            resultados.append(partido)

print(resultados)
