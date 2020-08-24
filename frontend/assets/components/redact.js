import { getComments, getTicketId, requestRedactApi } from "../apis/zafApis.js";
import { removeWhitespace } from "../utils.js";
import { entitiesStringSplitDelimiter } from "../constants.js";
import client from "../apis/zafClient.js";

const redact = async (text) => {
  const entities = text.split(entitiesStringSplitDelimiter).map(removeWhitespace);

  const ticketId = await getTicketId();
  const comments = await getComments();

  for (const comment of comments) {
    for (const entity of entities) {
      if (comment["text"].includes(entity)) {
        await requestRedactApi(ticketId, comment["id"], entity);
      }
    }
  }

  client.invoke("notify", "Redaction completed!");
};

export default redact;
