export const checkInfluentialWords = async (noticia) => {
  try {
    const response = await fetch("http://127.0.0.1:8000/check_important_words", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        title: noticia.Titulo,
        description: noticia.Descripcion,
      }),
    });

    if (!response.ok) {
      throw new Error("Error al obtener palabras influyentes");
    }
    const data = await response.json();
    
    console.log("Respuesta de palabras influyentes:", data);
    return data.important_words_found || [];
  } catch (error) {
    console.error("Error en checkInfluentialWords:", error);
    return [];
  }
};
export const predictFakeNews = async (noticias) => {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ noticias }),
  });

  const data = await response.json();
  return data.resultados;
};
export const getModelMetrics = async () => {
  const response = await fetch("http://127.0.0.1:8000/model-metrics");
  const data = await response.json();
  return data;
};
export const getTopWords = async () => {
  const response = await fetch("http://127.0.0.1:8000/top-words");
  const data = await response.json();
  return data;
};