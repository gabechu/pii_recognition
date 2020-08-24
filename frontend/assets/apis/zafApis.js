import client from "./zafClient.js";

const getComments = async () => {
  const response = await client.get("ticket.comments");
  const comments = response["ticket.comments"];
  const commentsIdValue = comments.map((comment) => ({
    id: comment.id,
    text: comment.value,
  }));

  return commentsIdValue;
};

const getCommentTexts = async () => {
  const comments = await getComments();
  // note text contains HTML
  const commentTexts = comments.map((comment) => comment["text"]);
  return commentTexts;
};

const getTicketId = async () => {
  const response = await client.get("ticket.id");
  return response["ticket.id"];
};

const requestRedactApi = async (ticketId, commentId, text) => {
  let options = {
    url: `/api/v2/tickets/${ticketId}/comments/${commentId}/redact.json`,
    type: "PUT",
    contentType: "application/json",
    data: JSON.stringify({ text: `${text}` }),
  };

  // 400 error here means text not found
  await client.request(options);
};

export { getCommentTexts, getComments, getTicketId, requestRedactApi };
