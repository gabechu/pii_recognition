import getPiiEntities from "./piiDetect.js";

// must take no argument
const onClickDetect = async () => {
  const modelName = document.querySelector("#model-name").value;
  const piiEntities = await getPiiEntities(modelName);

  document.querySelector("#pii-entities").value = piiEntities;
};

document.querySelector("#detect").addEventListener("click", onClickDetect);
