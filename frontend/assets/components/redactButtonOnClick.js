import redact from "./redact.js";

const onClickRedact = () => {
  const text = document.querySelector("#pii-entities").value;
  redact(text);
};

document.querySelector("#redact").addEventListener("click", onClickRedact);
