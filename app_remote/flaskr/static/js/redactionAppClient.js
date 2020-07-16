/** A client communicates with redaction Flask app. */
const RedactionAppClient = class {
  constructor (host) {
    this.host = host
  }

  /**
   * Send ticket id to redaction app for a replacement.
   * @param {object} payload An object has key `ticket.id`.
   * @return {Promise}
   */
  async replaceTicketId (payload) {
    const jsonifyBody = JSON.stringify(payload['ticket.id'])

    return fetch(`${this.host}/replace-ticket-id`, {
      method: 'PUT',
      body: jsonifyBody,
      cache: 'no-cache',
      headers: {
        'Content-Type': 'application/json'
      }
    })
  }
}

export { RedactionAppClient }
